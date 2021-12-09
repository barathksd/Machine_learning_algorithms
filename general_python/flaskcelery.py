# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:24:29 2021

@author: 81807
"""

from flask import Flask
from celeryconfig import make_celery

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='pyamqp://guest@localhost:5672//',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)
celery = make_celery(app)

@celery.task(name='celeryconfig.add')
def add_together(a, b):
    return a + b

@app.route('/')
def run():
    add_together.delay(5,6)
    return 'running'

if __name__ == '__main__':
    app.run(debug=True)



