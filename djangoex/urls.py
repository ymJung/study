from django.conf.urls import patterns, include, url
from student.views import *
from personal.views import *

from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'apiSrv.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),

	
    url(r'^student/$', student_view),
    url(r'^student/(?P<id>\d+)/$', student_detail_view),
    url(r'^personal/$', personal_view),

)
