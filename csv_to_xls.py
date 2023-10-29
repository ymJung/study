
fn = ''
csv_lines = open(fn + '.csv').readlines()

from openpyxl import Workbook

workbook = Workbook()
sheet = workbook.active

headers = csv_lines[0].split(',')

def combine(datas):
    result = ''
    for data in datas:
        result += data + ','
    return result[0:result.rfind(',')]


for idx in range(len(csv_lines)):
    csv_line = csv_lines[idx]
    row_num = (idx + 1)
    datas = csv_line.split(',')
    for jdx in range(len(headers)):
        col_num = (jdx + 1)
        data = datas[jdx]
        if col_num is len(headers):
            data = combine(datas[jdx:len(datas)])
            

        sheet.cell(row_num, col_num, data)

workbook.save(fn + '.xlsx')