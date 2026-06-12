#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:35:04 2026

@author: daniel
"""

from NEXT.interactive import cmdrun, gui
import sys

if len(sys.argv) > 1 and sys.argv[1] == "--gui":
    gui()
else:
    cmdrun()