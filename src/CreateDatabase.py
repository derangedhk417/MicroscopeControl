import os
import sys
import code
import time
import json

import numpy             as np
import matplotlib.pyplot as plt

import sqlite3 as sql

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("This program expects two arguments, the directory to process and the maximum ")
		print("number of layers to consider in the database.")
		exit()

	path = sys.argv[1]
	N    = int(sys.argv[2])

	# Create a database in the target directory.
	dbname = os.path.join(path, "_database.db")
	con    = sql.connect(dbname)
	cur    = con.cursor()

	# Now we prepare a create statement that will generate a table to store all of this information
	# in.
	layer_columns = ['L%03d_area REAL'%(i + 1) for i in range(N)]
	column_spec   = "(id INT PRIMARY KEY, file TEXT, geom_idx INT, area REAL, %s)"%(", ".join(layer_columns))

	cur.execute("CREATE TABLE flakes %s"%column_spec)

	# Now we read through every file in the directory that ends with .stats.json and insert every
	# entry into the database.

	files = []
	for entry in os.listdir(path):
		file = entry.replace("\\", "/").split("/")[-1]
		ext  = ".".join(file.split(".")[-2:])
		if ext == "stats.json":
			files.append(os.path.join(path, entry))

	current_id = 0
	for i, file in enumerate(files):
		with open(file, 'r') as f:
			data = json.loads(f.read())

		for flake in data['flakes']:
			columns       = "id, %s"%(", ".join(flake.keys()))
			value_strings = []
			for v in flake.values():
				if type(v) is str:
					value_strings.append("'%s'"%v)
				else:
					value_strings.append("%.2f"%float(v))

			values    = "%d, %s"%(current_id, ", ".join(value_strings))
			statement = "INSERT INTO flakes (%s) VALUES (%s)"%(columns, values)
			cur.execute(statement)
			current_id += 1

		print("file %d / %d"%(i + 1, len(files)))

	con.commit()
	con.close()