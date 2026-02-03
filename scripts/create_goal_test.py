import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE','expensetracker.settings')
import django
django.setup()
from django.contrib.auth.models import User
from goals.models import Goal
u, created = User.objects.get_or_create(username='testuser', defaults={'email':'test@example.com'})
if created:
    u.set_password('testpass')
    u.save()
print('User:', u.username)
g = Goal(name='Test Goal', owner=u, start_date='2026-01-01', end_date='2026-12-31', amount_to_save=100.00)
g.save()
print('Goal saved with id', g.id)
