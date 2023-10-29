tmp = ''' '''
datas = tmp.split('\n')
results = []
for data in datas:
    result = ''
    pos = 0
    for idx in range(len(data)):        
        pos += 1
        if pos >= len(data):
            break
        compare = data[pos]
        if compare == '_':
            pos += 1
            if pos > len(data):
                break
            result += data[pos].upper()
        else :
            result += data[pos]
    results.append(result)

for result in results:
    print(result)