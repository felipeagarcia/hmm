# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:48:16 2017

@author: AMD
"""

import csv

class csvHandler():
    def __init__(self, name):
        self.name = name;
        
    def save(self, string, tdelimiter):
        aux_str = string.split(tdelimiter);
        with open (self.name + ".csv", 'a', newline = '') as csvfile:
            writer =  csv.writer(csvfile)
            writer.writerow((aux_str[0],aux_str[1],aux_str[2],aux_str[3]))
        
    def load(self):
        return 1