from flask import Flask, request, render_template
import pickle

file = open("body_fat_model.pkl", "rb")
rf = pickle.load(file)
file.close()