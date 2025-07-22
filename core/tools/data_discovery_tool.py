from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.agent_tool import AgentTool
from core.subagents import create_extract_source_datasource_agent

from core.utils import DataProfiler,DataSourceConnector,DataSourceReport,TableProfile,DataSourceConfig
import logging
import datetime
import yaml
import json
from dataclasses import asdict
from typing import Dict, List, Any


class DataDiscoveryTool(BaseTool):
    """ADK Tool for data discovery operations"""

    def __init__(self):
        super().__init__(
            name="data_discovery",
            description="Performs comprehensive data discovery and profiling for healthcare payer data sources"
        )
        self.profiler = DataProfiler()
        self.connector = DataSourceConnector(self.profiler)
        #self.requirement_reader = RequirementFileReader()

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:

        agent_tool = AgentTool(agent=create_extract_source_datasource_agent())
        ds_agent_output = await agent_tool.run_async(
            args={"file_path": args['file_path']}, tool_context=tool_context
        )
        return self.run(yaml_requirement_content=ds_agent_output.text)


    async def run(self, yaml_requirement_content: str, output_format: str = "json") -> str:
        """Execute data discovery process"""
        try:
            # Read requirements
            #if requirement_file.endswith('.yaml') or requirement_file.endswith('.yml'):
            configs = self.read_yaml_requirements(yaml_requirement_content)
            reports = []

            for config in configs:
                report = await self._discover_data_source(config)
                reports.append(report)

            # Generate consolidated report
            consolidated_report = self._generate_consolidated_report(reports)

            if output_format.lower() == "json":
                return json.dumps(asdict(consolidated_report), indent=2, default=str)
            else:
                return self._generate_html_report(consolidated_report)

        except Exception as e:
            return f"Error during data discovery: {str(e)}"

    def read_yaml_requirements(yaml_content: str) -> List[DataSourceConfig]:
        """Read YAML requirements file"""
        data = yaml.safe_load(yaml_content)

        configs = []
        for source in data.get('data_sources', []):
            config = DataSourceConfig(
                name=source['name'],
                type=source['type'],
                connection_params=source['connection_params'],
                tables_or_files=source['tables_or_files'],
                description=source.get('description')
            )
            configs.append(config)

        return configs

    async def _discover_data_source(self, config: DataSourceConfig) -> DataSourceReport:
        """Discover and profile a single data source"""
        table_profiles = []
        total_rows = 0
        total_size_mb = 0.0
        phi_columns = []

        for table_or_file in config.tables_or_files:
            try:
                # Read data based on source type
                if config.type == 'postgresql':
                    data = await self.connector.read_postgresql(config.connection_params, table_or_file)
                elif config.type == 'mysql':
                    data = await self.connector.read_mysql(config.connection_params, table_or_file)
                elif config.type == 'mongodb':
                    data = await self.connector.read_mongodb(config.connection_params, table_or_file)
                elif config.type == 'csv':
                    data = await self.connector.read_csv_file(table_or_file)
                elif config.type == 'parquet':
                    data = await self.connector.read_parquet_file(table_or_file)
                elif config.type == 'json':
                    data = await self.connector.read_json_file(table_or_file)


                # Profile the data
                profile = await self.profiler.profile_table(data, table_or_file)
                table_profiles.append(profile)

                total_rows += profile.row_count
                total_size_mb += profile.estimated_size_mb

                # Collect PHI columns
                for col_profile in profile.columns:
                    if col_profile.is_potential_phi:
                        phi_columns.append(f"{table_or_file}.{col_profile.column_name}")

            except Exception as e:
                logging.error(f"Error processing {table_or_file}: {e}")
                continue

        # Generate recommendations
        recommendations = self._generate_recommendations(table_profiles, phi_columns)

        # Calculate data quality summary
        quality_summary = self._calculate_quality_summary(table_profiles)

        return DataSourceReport(
            source_name=config.name,
            source_type=config.type,
            total_tables=len(table_profiles),
            total_rows=total_rows,
            total_size_mb=round(total_size_mb, 2),
            table_profiles=table_profiles,
            phi_columns_detected=phi_columns,
            data_quality_summary=quality_summary,
            recommendations=recommendations,
            generated_at=datetime.now()
        )

    def _generate_recommendations(self, profiles: List[TableProfile], phi_columns: List[str]) -> List[str]:
        """Generate data governance and quality recommendations"""
        recommendations = []

        if phi_columns:
            recommendations.append(
                f"⚠️  {len(phi_columns)} potential PHI columns detected. Implement data masking and access controls.")

        low_quality_tables = [p for p in profiles if p.data_quality_score < 70]
        if low_quality_tables:
            recommendations.append(
                f"📊 {len(low_quality_tables)} tables have data quality scores below 70%. Consider data cleansing.")

        large_tables = [p for p in profiles if p.row_count > 1000000]
        if large_tables:
            recommendations.append(f"🔄 {len(large_tables)} large tables detected. Consider partitioning strategies.")

        high_null_columns = []
        for profile in profiles:
            for col in profile.columns:
                if col.null_percentage > 50:
                    high_null_columns.append(f"{profile.name}.{col.column_name}")

        if high_null_columns:
            recommendations.append(
                f"🚫 {len(high_null_columns)} columns have >50% null values. Review data collection processes.")

        return recommendations

    def _calculate_quality_summary(self, profiles: List[TableProfile]) -> Dict[str, Any]:
        """Calculate overall data quality summary"""
        if not profiles:
            return {}

        avg_quality_score = sum(p.data_quality_score for p in profiles) / len(profiles)
        total_columns = sum(p.column_count for p in profiles)
        phi_columns = sum(1 for p in profiles for c in p.columns if c.is_potential_phi)

        return {
            "average_quality_score": round(avg_quality_score, 2),
            "total_columns_analyzed": total_columns,
            "potential_phi_columns": phi_columns,
            "tables_analyzed": len(profiles)
        }

    def _generate_consolidated_report(self, reports: List[DataSourceReport]) -> Dict[str, Any]:
        """Generate consolidated discovery report"""
        return {
            "summary": {
                "total_sources": len(reports),
                "total_tables": sum(r.total_tables for r in reports),
                "total_rows": sum(r.total_rows for r in reports),
                "total_size_mb": sum(r.total_size_mb for r in reports),
                "total_phi_columns": sum(len(r.phi_columns_detected) for r in reports)
            },
            "source_reports": reports,
            "generated_at": datetime.now()
        }

