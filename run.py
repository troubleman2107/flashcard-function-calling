#!/usr/bin/env python3
"""
Simple run script for the flashcard application
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from app import app

    print("Starting Flashcard Application...")
    print("Visit http://localhost:6969 to use the application")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host="0.0.0.0", port=6969)
