# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:07:52 2024

@author: Joao

This file is currently unused. It is kept here since reading from the database
maybe be several times faster and simpler (not require any output files)

"""

import sqlite3
db_file = './P2Ms/simple_street_canyon/study_rays=0.25_res=2m_3ghz/simple_street_canyon_test.study_rays=0.25_res=2m_3ghz.sqlite'
# Connect to the SQLite database
connection = sqlite3.connect(db_file)

# Create a cursor
cursor = connection.cursor()

# Get the list of all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in the database:")
for table in tables:
    print(f" - {table[0]}")

# Get the schema for each table
for table in tables:
    table_name = table[0]
    print(f"\nSchema for table '{table_name}':")
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()
    for column in schema:
        print(column)

# Close the connection
connection.close()
