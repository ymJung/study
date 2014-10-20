from django.db import models
from django.contrib import admin


# Create your models here.
class Student(models.Model):
    name = models.CharField(max_length=30)
    age  = models.IntegerField()


class StudentAdmin(admin.ModelAdmin):
    list_display = ["name", "age"]
    search_fields = ["name"]

