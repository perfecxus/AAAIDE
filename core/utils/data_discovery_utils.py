from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy import create_engine, inspect


import pandas as pd
import pymongo
import logging
import re



@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    name: str
    type: str  # postgresql, mysql, oracle, mssql, snowflake, iceberg, delta, mongodb, dynamodb, cosmosdb, file
    connection_params: Dict[str, Any]
    tables_or_files: List[str]
    description: Optional[str] = None


@dataclass
class ColumnProfile:
    """Data profile for a single column"""
    column_name: str
    data_type: str
    nullable: bool
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    std_dev: Optional[float] = None
    top_values: Optional[List[Tuple[Any, int]]] = None
    has_special_chars: bool = False
    is_potential_phi: bool = False
    date_format: Optional[str] = None
    constraints: Optional[List[str]] = None
    bounded_values: Optional[List[Any]] = None


@dataclass
class TableProfile:
    """Data profile for a table or file"""
    name: str
    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    data_quality_score: float
    estimated_size_mb: float
    last_updated: Optional[datetime] = None
    partitions: Optional[List[str]] = None


@dataclass
class DataSourceReport:
    """Complete data discovery report for a data source"""
    source_name: str
    source_type: str
    total_tables: int
    total_rows: int
    total_size_mb: float
    table_profiles: List[TableProfile]
    phi_columns_detected: List[str]
    data_quality_summary: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime


class PHIDetector:
    """Detects potential PHI (Protected Health Information) in healthcare data"""

    def __init__(self):
        # Common PHI patterns
        self.phi_patterns = {
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'mrn': r'\b(MRN|mrn)\s*:?\s*\d+\b',
            'member_id': r'\b(MEMBER|MBR|ID)\s*:?\s*[A-Z0-9]{6,}\b',
            'date_of_birth': r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b'
        }

        # PHI column name patterns
        self.phi_column_names = {
            'patient_name', 'first_name', 'last_name', 'full_name',
            'ssn', 'social_security', 'member_id', 'patient_id',
            'phone_number', 'telephone', 'mobile', 'email',
            'address', 'street', 'city', 'zip', 'zipcode',
            'date_of_birth', 'dob', 'birth_date',
            'medical_record_number', 'mrn', 'account_number'
        }

    def is_potential_phi_column(self, column_name: str, sample_data: List[str]) -> bool:
        """Check if a column might contain PHI based on name and sample data"""
        column_lower = column_name.lower().replace('_', '').replace('-', '')

        # Check column name patterns
        for phi_name in self.phi_column_names:
            if phi_name.replace('_', '') in column_lower:
                return True

        # Check data patterns
        if sample_data:
            for pattern_name, pattern in self.phi_patterns.items():
                for sample in sample_data[:100]:  # Check first 100 samples
                    if sample and re.search(pattern, str(sample)):
                        return True

        return False


# class TrinoMCPConnector:
#     """Connector to Trino via MCP server for multi-database support"""
#
#     def __init__(self, trino_host: str = "localhost", trino_port: int = 8080):
#         self.trino_host = trino_host
#         self.trino_port = trino_port
#         self.connection = None
#
#     async def connect(self) -> bool:
#         """Establish connection to Trino"""
#         try:
#             self.connection = trino.dbapi.connect(
#                 host=self.trino_host,
#                 port=self.trino_port,
#                 user="discovery_agent"
#             )
#             return True
#         except Exception as e:
#             logging.error(f"Failed to connect to Trino: {e}")
#             return False
#
#     async def execute_query(self, query: str, catalog: str = None) -> pd.DataFrame:
#         """Execute query via Trino and return results as DataFrame"""
#         if not self.connection:
#             await self.connect()
#
#         try:
#             cursor = self.connection.cursor()
#             if catalog:
#                 cursor.execute(f"USE {catalog}")
#
#             cursor.execute(query)
#             columns = [desc[0] for desc in cursor.description]
#             rows = cursor.fetchall()
#
#             return pd.DataFrame(rows, columns=columns)
#         except Exception as e:
#             logging.error(f"Query execution failed: {e}")
#             return pd.DataFrame()
#

class DataProfiler:
    """Main data profiling engine"""

    def __init__(self):
        self.phi_detector = PHIDetector()
       # self.trino_connector = TrinoMCPConnector()

    async def profile_column(self, data: pd.Series, column_name: str) -> ColumnProfile:
        """Profile a single column"""
        total_count = len(data)
        null_count = data.isnull().sum()
        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0

        unique_count = data.nunique()
        unique_percentage = (unique_count / total_count) * 100 if total_count > 0 else 0

        # Get sample data for PHI detection
        sample_data = data.dropna().astype(str).head(100).tolist()

        profile = ColumnProfile(
            column_name=column_name,
            data_type=str(data.dtype),
            nullable=null_count > 0,
            null_count=int(null_count),
            null_percentage=round(null_percentage, 2),
            unique_count=int(unique_count),
            unique_percentage=round(unique_percentage, 2),
            is_potential_phi=self.phi_detector.is_potential_phi_column(column_name, sample_data)
        )

        # Numeric statistics
        if pd.api.types.is_numeric_dtype(data):
            profile.min_value = data.min()
            profile.max_value = data.max()
            profile.mean_value = float(data.mean()) if not pd.isna(data.mean()) else None
            profile.std_dev = float(data.std()) if not pd.isna(data.std()) else None

        # Top values (for categorical data)
        if unique_count <= 20:  # Only for low cardinality columns
            top_values = data.value_counts().head(10)
            profile.top_values = [(val, int(count)) for val, count in top_values.items()]
            profile.bounded_values = list(data.unique())

        # Special character detection
        if data.dtype == 'object':
            profile.has_special_chars = any(
                bool(re.search(r'[^\w\s]', str(val))) for val in sample_data[:50]
            )

        # Date format detection
        if 'date' in column_name.lower() or data.dtype == 'datetime64[ns]':
            profile.date_format = self._detect_date_format(sample_data[:10])

        return profile

    def _detect_date_format(self, sample_dates: List[str]) -> Optional[str]:
        """Detect common date formats"""
        formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S', '%Y-%m-%dT%H:%M:%S'
        ]

        for fmt in formats:
            try:
                for date_str in sample_dates:
                    if date_str:
                        datetime.strptime(str(date_str), fmt)
                return fmt
            except:
                continue
        return None

    async def profile_table(self, data: pd.DataFrame, table_name: str) -> TableProfile:
        """Profile an entire table/dataset"""
        row_count = len(data)
        column_count = len(data.columns)

        # Profile each column
        column_profiles = []
        for column in data.columns:
            profile = await self.profile_column(data[column], column)
            column_profiles.append(profile)

        # Calculate data quality score
        quality_score = self._calculate_quality_score(column_profiles, row_count)

        # Estimate size
        estimated_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)

        return TableProfile(
            name=table_name,
            row_count=row_count,
            column_count=column_count,
            columns=column_profiles,
            data_quality_score=round(quality_score, 2),
            estimated_size_mb=round(estimated_size_mb, 2)
        )

    def _calculate_quality_score(self, column_profiles: List[ColumnProfile], row_count: int) -> float:
        """Calculate overall data quality score (0-100)"""
        if not column_profiles:
            return 0.0

        total_score = 0.0

        for profile in column_profiles:
            column_score = 100.0

            # Penalize high null percentages
            if profile.null_percentage > 50:
                column_score -= 30
            elif profile.null_percentage > 20:
                column_score -= 15
            elif profile.null_percentage > 10:
                column_score -= 5

            # Reward high uniqueness for ID columns
            if 'id' in profile.column_name.lower() and profile.unique_percentage < 90:
                column_score -= 20

            # Penalize very low uniqueness (potential data issues)
            if profile.unique_percentage < 1:
                column_score -= 10

            total_score += column_score

        return total_score / len(column_profiles)


class DataSourceConnector:
    """Handles connections to various data sources"""

    def __init__(self, profiler: DataProfiler):
        self.profiler = profiler

    async def read_postgresql(self, config: Dict[str, Any], table: str) -> pd.DataFrame:
        """Read data from PostgreSQL"""
        engine = create_engine(
            f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}")
        return pd.read_sql(f"SELECT * FROM {table} LIMIT 10000", engine)

    async def read_mysql(self, config: Dict[str, Any], table: str) -> pd.DataFrame:
        """Read data from MySQL"""
        engine = create_engine(
            f"mysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}")
        return pd.read_sql(f"SELECT * FROM {table} LIMIT 10000", engine)

    async def read_mongodb(self, config: Dict[str, Any], collection: str) -> pd.DataFrame:
        """Read data from MongoDB"""
        client = pymongo.MongoClient(config['connection_string'])
        db = client[config['database']]
        cursor = db[collection].find().limit(10000)
        return pd.DataFrame(list(cursor))

    async def read_csv_file(self, file_path: str) -> pd.DataFrame:
        """Read CSV file"""
        return pd.read_csv(file_path)

    async def read_parquet_file(self, file_path: str) -> pd.DataFrame:
        """Read Parquet file"""
        return pd.read_parquet(file_path)

    async def read_json_file(self, file_path: str) -> pd.DataFrame:
        """Read JSON file"""
        return pd.read_json(file_path)