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












