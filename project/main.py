# -*- coding: utf-8 -*-

'''Entry point to all things to avoid circular imports.'''
from app import freezer, pages, server
# from app import server as app
from project.views import *
