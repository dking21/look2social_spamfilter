from flask import Flask, request, render_template, jsonify
import json
import requests
import os
import pymongo
import time
import random
import pandas as pd
import pickle
import webapi
import numpy as np

app = Flask(__name__, static_url_path="")

df = pd.read_excel('website_display.xlsx')

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')

@app.route('/get_results', methods=['GET'])
def show_results(index_num):
    return df[df['index'] == index_num]['text']
