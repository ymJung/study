from django.shortcuts import render
from django.http import HttpResponse

import json

# Create your views here.

def toJSON(objs, status=200) :
   j = json.dumps(objs, ensure_ascii=False)
   return HttpResponse(j, status=status, content_type='applcation/json; charset=utf-8')

def student_view(request) :
    sts = Student.objects.all()

    rst = {}
    rst['name'] = sts[0].name
    rst['age'] = sts[0].age

    return toJSON(rst)

def student_detail_view(request, id) :
    rst = {}
    rst['msg'] = "hello"

    return toJSON(rst)
