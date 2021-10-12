# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:15:52 2021

@author: 81807
"""

from flask_wtf import FlaskForm
from datetime import date
from wtforms import StringField, DateTimeField, SubmitField, DateField, TimeField
from wtforms.validators import DataRequired, Length
from __init__ import is_validname

class SearchForm(FlaskForm):
    name = StringField('Name',validators=[DataRequired(),Length(min=1,max=21)],id='sname')
    sdate = DateField('Date',default=date.today(),id='sdate')
    stime = TimeField('Start Time',id='stime')
    etime = TimeField('End Time',id='etime')
    
    submit = SubmitField('Submit')
    def validate_on_submit(self):
        result = super(SearchForm, self).validate()
        if not is_validname(self.name.data):
            return False
        return result
    
    
    