#!/usr/bin/env python3
"""
CoT Reasoning Generator Launcher
==============================

Simple launcher script for the Chain of Thought reasoning generator.
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from cot_maker import main

if __name__ == "__main__":
    main()