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


# Query to get all XYZ positions of the receivers
cursor.execute('SELECT x, y, z FROM rx')  # Replace 'receivers' with your actual table name

# Fetch all results
xyz_positions = cursor.fetchall()

# Print the results
print("XYZ Positions of Receivers:")
for position in xyz_positions:
    print(f"X: {position[0]}, Y: {position[1]}, Z: {position[2]}")


cursor.execute('SELECT rx_id FROM rx')
rx_ids = cursor.fetchall()

# Close the connection
connection.close()

#%%
import matplotlib.pyplot as plt
xyz = np.array(xyz_positions)

rx_ids = np.array(rx_ids)

plt.figure(dpi=200, )
plt.scatter(xyz[:,0], xyz[:,1], s=2)
plt.ylim((-70,70))
plt.xlim((-100, 100))
plt.show()


#%%
fig = plt.figure(dpi=200) 
ax = fig.add_subplot(projection='3d')
n = 80
k = 8
ax.scatter(xs=xyz[:n:k,0], ys=xyz[:n:k,1], zs=rx_ids[:n:k], s=1)
ax.set_ylim((-70,70))
ax.set_xlim((-100, 100))
plt.show()

#%%

plt.plot(rx_ids)

#%%

"""
There are two problems in the database:
    - The indices are not continuous
    - Seems to have more positions than the ones we actually simulates..
    (5799 vs 5551 = 61*91)
"""
