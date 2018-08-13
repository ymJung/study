file_name = 'conn.xml'
FILE_EXPORT = True
passwd = input('passwd?')
port = 3306
GROUP = input('group?')
SKIP_SCH = [""]
QUERIES = []




import xml.etree.ElementTree
el = xml.etree.ElementTree.parse(file_name)

conn_infos = []
connes = []
for it in el.getroot().iter('Group'):
    g_name = it.attrib['name']
    if GROUP == g_name:
        connes.append(it)
for each in connes:
    for it in each.iter('Connection'):
        conn_info = {}
        if it.find('Database').text not in SKIP_SCH:
            conn_info['schme'] = it.find('Database').text
            conn_info['host'] = it.find('Host').text
            conn_info['id'] = it.find('User').text
            conn_info['passwd'] = passwd
            conn_info['port'] = port
            conn_infos.append(conn_info)
def print_results(results):
    for result in results:
        print(result)

    if FILE_EXPORT :
        f = open(str(strftime("FILE_EXPORT", localtime())) + ".txt", 'a')
        for result in results:
            for k, val in result.items():
                f.write(str(val) + '\t')
            f.write('\n')
        f.close()



import pymysql
from time import localtime, strftime, sleep

for conn_info in conn_infos:
    conn = pymysql.connect(host=conn_info['host'], port=conn_info['port'], user=conn_info['id'], passwd=conn_info['passwd'], db=conn_info['schme'], charset='utf8', cursorclass=pymysql.cursors.DictCursor)
    print(conn_info['host'], conn_info['schme'])
    for query in QUERIES:
        if len(query) == 0 or '--' in query:
            continue
        cursor = conn.cursor()
        if query.strip().lower().startswith('select') :
            cursor.execute(query)
            results = cursor.fetchall()
            print_results(results)


        if query.strip().lower().startswith('update') or query.strip().lower().startswith('insert'):
            #sel = input('['+conn_info['host']+'] [' + conn_info['schme']+'] ['+query+'] go?(N or ENTER)')
            sel = ''
            if sel.lower() != 'n':
                print('\t' + query)
                cursor.execute(query)
                conn.commit()
    conn.close()
print('done')
