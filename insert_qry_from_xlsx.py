import xlrd

HEADER_ROW = 0
file_name = '201104prd.xlsx'
SKIP_VALUE = ['SYSDATE', 'NULL']
SKIP_NUMBER = []
workbook = xlrd.open_workbook(file_name)
sheet = workbook.sheet_by_index(0)



SKIP_COL_NUM = []

def is_number(input_value):
    try :
        float(input_value)
        return True
    except ValueError:
        return False

def start_with_select(input_value):
    if is_number(input_value) is False:
        return input_value.find('(SELECT') >= 0
    return False


def attatch_text(xls_input, col_num):
    result = ''
    if col_num in SKIP_COL_NUM:
        result = str(xls_input)
    else:
        if xls_input in SKIP_VALUE:            
            result = str(xls_input)
        elif is_number(xls_input):
            try :
                result = str(round(xls_input))   
            except:
                result = str(xls_input)  
            
        elif start_with_select(xls_input):
            result = str(xls_input)        
        else :
            if "'" in xls_input:
                result = xls_input
            else :
                result = "'" + xls_input + "'"
    return result + ","
    


HEADER_COL_LEN = 0

for cell in sheet.row(HEADER_ROW):
    if cell.value is not '':
        HEADER_COL_LEN += 1

header_qry = 'INSERT INTO ' + sheet.name + '('
for col in range(HEADER_COL_LEN):
    header_qry += sheet.cell_value(HEADER_ROW, col) + ', '
# headers 
header_qry = header_qry[0:header_qry.rfind(',')] + ') VALUES '

insert_qry = ''
for row in range((HEADER_ROW + 1), sheet.nrows):
    for col in range(HEADER_COL_LEN):
        if col is 0:
            insert_qry += header_qry + '('
        insert_qry += attatch_text(sheet.cell_value(row, col), col)
        if col is HEADER_COL_LEN - 1:
            insert_qry = insert_qry[0:insert_qry.rfind(',')]
            insert_qry += ');\n'


f = open(file_name + '.sql', 'w')
f.write(insert_qry)
f.close()


