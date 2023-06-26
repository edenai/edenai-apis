"""
Monitoring tool, allowing to track all the calls you made and insert them in db.
This is totally optional and can be opted in with the `MONITORING` environment variable

To use this feature the db need to be created first,
You can then use the following snippet to create the db

```sql
CREATE TABLE IF NOT EXISTS history (
         provider varchar(100),
         feature varchar(100),
         subfeature varchar(100),
         start_date timestamp,
         edenai_user varchar(100),
         cost numeric(15, 10),
         unit_type varchar(100),
         nb_unit integer,
         environment varchar(100),
         host varchar(100),
         host_user varchar(100),
         error varchar(255)
     );
     GRANT INSERT ON TABLE history TO history_write_only;
```
"""
import getpass
import os
import socket
from datetime import datetime
from typing import Optional

import psycopg2
from loaders.data_loader import ProviderDataEnum
from loaders.loaders import load_provider
from psycopg2 import errors
from psycopg2.extensions import AsIs

from .upload_s3 import get_providers_json_from_s3

global INFOS_FROM_S3
global POSTGRES_CONNECTION


def monitor_call(condition=False):
    """decorator for compute output functions to add monitoring features"""
    def decorator_monitor_call(compute_func):
        def wrapper(
            provider_name,
            feature,
            subfeature,
            *args,
            **kwargs,
        ):
            fake = kwargs.get("fake", False)
            error = "Fake" if fake else None
            user_email= kwargs.get("user_email")
            try:
                 return compute_func(
                     provider_name,
                     feature,
                     subfeature,
                    *args,
                    **kwargs,
                )
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                if condition:
                    insert_api_call(
                        provider=provider_name,
                        feature=feature,
                        subfeature=subfeature,
                        user_email=user_email,
                        error=error,
                    )
        return wrapper
    return decorator_monitor_call


def insert_api_call(
    provider: str,
    feature: str,
    subfeature: str,
    user_email: Optional[str],
    error: Optional[str],
):
    global POSTGRES_CONNECTION
    if not "POSTGRES_CONNECTION" in globals():
        rds_settings = load_provider(ProviderDataEnum.KEY, "rds")
        # Connect to your postgres DB
        POSTGRES_CONNECTION = psycopg2.connect(
            f"dbname=history_db user={rds_settings['write_only_user']} "
            + f"password={rds_settings['write_only_password']} host={rds_settings['host']}"
        )
        print("Connect to postgres history ok")
    global INFOS_FROM_S3
    if not "INFOS_FROM_S3" in globals():
        print("Download providers price from s3")
        INFOS_FROM_S3 = get_providers_json_from_s3()

    to_insert = {
        "provider": provider,
        "feature": feature,
        "subfeature": subfeature,
        "environment": os.environ.get(
            "GIT_BRANCH", os.environ.get("CIRCLE_BRANCH", "local_dev")
        ),
        "host": os.environ.get("HOSTNAME", socket.gethostname()),
        "start_date": datetime.utcnow(),
        "edenai_user": user_email,
        "error": error,
        "host_user": getpass.getuser(),
    }

    columns = to_insert.keys()
    values = [to_insert[column] for column in columns]

    insert_statement = "insert into history (%s) values %s"

    # Open a cursor to perform database operations
    try:
        cur = POSTGRES_CONNECTION.cursor()
        cur.execute(insert_statement, (AsIs(",".join(columns)), tuple(values)))
        POSTGRES_CONNECTION.commit()  # <--- makes sure the change is shown in the database
    except errors.InFailedSqlTransaction as exc:
        print(exc)
        POSTGRES_CONNECTION.rollback()

    cur.close()
