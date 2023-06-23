from flask import Flask, request, render_template
import pickle

file = open("body_fat_model.pkl", "rb")
rf = pickle.load(file)
file.close()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        my_dict = request.form

        density = float(my_dict["density"])
        abdomen =float(my_dict["abdomen"])
        chest = float(my_dict["chest"])
        weight = float(my_dict["weight"])
        hip = float(my_dict["hip"])

        input_features = [[density, abdomen, chest, weight, hip]]
        prediction = rf.predict(input_features)[0].round(2)

        # <p class="big-font">Hello World !!</p>', unsafe_allow_html=True

        string = "Percentage of Body Fat Estimated is: " + str(prediction)+"%"

        return render_template("show.html", string=string)
    
if __name__=="__main__":
    app.run(debug=True)