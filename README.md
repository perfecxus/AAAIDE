# AAIDE

## Overview

### Remote agents are registered

<img width="1440" height="707" alt="image" src="https://github.com/user-attachments/assets/4b85bb2d-9213-4219-9d2a-4606b9dca1c3" />

### Same code snippet for execution logs 

```
(A2A-8bb8b364b387234d51a331705e47ba24abdf357c) rahulvishwakarma@Rahuls-MacBook-Air python % uv run agents/db_ingestion --port 11001
INFO:     Started server process [71737]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:11001 (Press CTRL+C to quit)
INFO:     ::1:58365 - "GET /.well-known/agent.json HTTP/1.1" 200 OK
```

## Example of A2A for Data Engineering agent
1. We are running http server on port 8000 

```
(A2A-8bb8b364b387234d51a331705e47ba24abdf357c) rahulvishwakarma@Rahuls-MacBook-Air A2A-Demo % python -m http.server 8000
Serving HTTP on :: port 8000 (http://[::]:8000/) ...
```

2. There are two .csv files 

```
(A2A-8bb8b364b387234d51a331705e47ba24abdf357c) rahulvishwakarma@Rahuls-MacBook-Air A2A-Demo % ls -ltr *.csv
-rw-r--r--@ 1 rahulvishwakarma  staff  92 Aug 18 03:11 jobs.csv
-rw-r--r--@ 1 rahulvishwakarma  staff  94 Aug 18 03:11 candidates.csv
```
3. Using A2A we will demonstrate the Data Ingestion Agent (port 11001)

<img width="1440" height="707" alt="image" src="https://github.com/user-attachments/assets/18d1e5b7-3021-4f29-aba3-fb587588ab5a" />

4. Event list

<img width="1440" height="457" alt="image" src="https://github.com/user-attachments/assets/3876bde8-926e-4bdd-9878-631be2894d21" />

## Agent cards

```
rahulvishwakarma@MacBookAir ~ % curl -s http://localhost:11000/.well-known/agent.json | jq .
{
  "name": "DB Health Agent",
  "description": "Operational health checks for PostgreSQL",
  "url": "http://localhost:11000/",
  "version": "1.0.0",
  "capabilities": {
    "streaming": false,
    "pushNotifications": false,
    "stateTransitionHistory": false
  },
  "defaultInputModes": [
    "text",
    "text/plain"
  ],
  "defaultOutputModes": [
    "text",
    "text/plain"
  ],
  "skills": [
    {
      "id": "db_health",
      "name": "Database Health",
      "description": "Checks PostgreSQL connectivity, version, schemas, table counts",
      "tags": [
        "postgres",
        "health",
        "ops"
      ],
      "examples": [
        "db version",
        "schema sizes",
        "tables by schema"
      ]
    }
  ]
}
rahulvishwakarma@MacBookAir ~ % 
rahulvishwakarma@MacBookAir ~ % curl -s http://localhost:11001/.well-known/agent.json | jq .
{
  "name": "Data Ingestion Agent",
  "description": "Ingests CSV into PostgreSQL",
  "url": "http://localhost:11001/",
  "version": "1.0.0",
  "capabilities": {
    "streaming": false,
    "pushNotifications": false,
    "stateTransitionHistory": false
  },
  "defaultInputModes": [
    "text",
    "text/plain"
  ],
  "defaultOutputModes": [
    "text",
    "text/plain"
  ],
  "skills": [
    {
      "id": "data_ingestion",
      "name": "Data Ingestion",
      "description": "Loads CSV from a URL into PostgreSQL",
      "tags": [
        "ingestion",
        "csv",
        "postgres"
      ],
      "examples": [
        "ingest csv https://example.com/employers.csv into workonward.raw_employers"
      ]
    }
  ]
}
rahulvishwakarma@MacBookAir ~ % curl -s http://localhost:11002/.well-known/agent.json | jq .
{
  "name": "Data Transform Agent",
  "description": "Simple SQL materializations",
  "url": "http://localhost:11002/",
  "version": "1.0.0",
  "capabilities": {
    "streaming": false,
    "pushNotifications": false,
    "stateTransitionHistory": false
  },
  "defaultInputModes": [
    "text",
    "text/plain"
  ],
  "defaultOutputModes": [
    "text",
    "text/plain"
  ],
  "skills": [
    {
      "id": "data_transform",
      "name": "Data Transform",
      "description": "Materializes cleaned tables and derived views",
      "tags": [
        "transform",
        "views",
        "etl"
      ],
      "examples": [
        "materialize cleaned candidates",
        "create jobs by city view"
      ]
    }
  ]
}
rahulvishwakarma@MacBookAir ~ % curl -s http://localhost:11003/.well-known/agent.json | jq .
{
  "name": "Data Quality Agent",
  "description": "Basic data quality checks",
  "url": "http://localhost:11003/",
  "version": "1.0.0",
  "capabilities": {
    "streaming": false,
    "pushNotifications": false,
    "stateTransitionHistory": false
  },
  "defaultInputModes": [
    "text",
    "text/plain"
  ],
  "defaultOutputModes": [
    "text",
    "text/plain"
  ],
  "skills": [
    {
      "id": "data_quality",
      "name": "Data Quality",
      "description": "Runs simple quality checks",
      "tags": [
        "dq",
        "quality"
      ],
      "examples": [
        "dq rowcount workonward.jobs",
        "dq nulls workonward.candidates email"
      ]
    }
  ]
}
rahulvishwakarma@MacBookAir ~ % curl -s http://localhost:11004/.well-known/agent.json | jq .
{
  "name": "SQL Analytics Agent",
  "description": "Read-only SQL execution",
  "url": "http://localhost:11004/",
  "version": "1.0.0",
  "capabilities": {
    "streaming": false,
    "pushNotifications": false,
    "stateTransitionHistory": false
  },
  "defaultInputModes": [
    "text",
    "text/plain"
  ],
  "defaultOutputModes": [
    "text",
    "text/plain"
  ],
  "skills": [
    {
      "id": "sql_analytics",
      "name": "SQL Analytics",
      "description": "Executes read-only SQL queries and returns tabular results",
      "tags": [
        "sql",
        "analytics"
      ],
      "examples": [
        "sql: select count(*) from workonward.jobs"
      ]
    }
  ]
}
rahulvishwakarma@MacBookAir ~ % 
```











