
import sqlite3
import os

conn = sqlite3.connect('signals.db')
c = conn.cursor()

try:
    c.execute('UPDATE signals SET outcome = ''EXPIRED'' WHERE outcome = ''OPEN''')
except Exception as e:
    pass

try:
    c.execute('UPDATE signal_outcomes SET outcome = ''EXPIRED'' WHERE outcome = ''OPEN''')
except Exception as e:
    pass

conn.commit()
conn.close()
print('تمت تصفية قاعدة البيانات بنجاح!')

