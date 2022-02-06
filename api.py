from flask import Flask,jsonify,request
from flask_cors import CORS,cross_origin

import pickle
import numpy as np

app = Flask(__name__)
CORS(app)
cors = CORS(app , resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})

model=pickle.load(open('model.pkl','rb'))
arr=[]
@app.route('/predict', methods=['POST'])
def predireCadeau():
    featuresReshaped=[]
    args={'arg1':request.json['arg1'],
    'arg2':request.json['arg2'],
    'arg3':request.json['arg3'],
    'arg4':request.json['arg4'],
    'arg5':request.json['arg5'],
    'arg6':request.json['arg6'],
    'arg7':request.json['arg7'],
    'arg8':request.json['arg8'],
    'arg9':request.json['arg9'],
    'arg10':request.json['arg10'],
    'arg11':request.json['arg11'],
    'arg12':request.json['arg12'],
    'arg13':request.json['arg13'],
    'arg14':request.json['arg14']
   
    }
    if (args['arg1']=="Femme"):
       arr.extend((1,0))
    else:
        arr.extend((0,1))
    if (args['arg2']=="Oui"):  
        arr.append(1)
    else:
        arr.append(0)
    if (args['arg3']=="Oui"):  
        arr.append(1)
    else:
        arr.append(0)
    if (args['arg4']=="Oui"):  
        arr.append(1)
    else:
        arr.append(0)
    if (args['arg5']=="Oui"):  
        arr.append(1)
    else:
        arr.append(0)
    if (args['arg6']=="Oui"):  
        arr.append(1)
    else:
        arr.append(0)
    if (args['arg7']=="Inférieur à 18 ans"):  
        arr.extend((0,0,1,0))
    elif (args['arg7']=="Entre 18 ans et 35 ans"):  
        arr.extend((1,0,0,0))
    elif (args['arg7']=="Entre 35 ans et 65 ans"):  
        arr.extend((0,1,0,0))
    elif (args['arg7']=="Supérieur à 65 ans"):  
        arr.extend((0,0,0,1))
    if (args['arg8']=="Pragmatique"):  
        arr.append(1)
    else:
        arr.append(0)
    if (args['arg9']=="Sérieux(se)"):  
        arr.append(1)
    else:
        arr.append(0)
    if (args['arg10']=="Introvertie"):  
        arr.append(0)
    else:
        arr.append(1)
    if (args['arg11']=="Anniversaire"):  
        arr.extend((1,0,0,0,0))
    elif (args['arg11']=="Graduation"):
        arr.extend((0,1,0,0,0))
    elif (args['arg11']=="Mariage"):
        arr.extend((0,0,1,0,0))
    elif (args['arg11']=="Nouvel-an"):
        arr.extend((0,0,0,1,0))
    elif (args['arg11']=="Evénement religieux"):
        arr.extend((0,0,0,0,1))
    elif (args['arg11']=="Evénement spécial"):
        arr.extend((0,0,0,0,0))
    if (args['arg12']=="< 100 DHs"):  
        arr.extend((0,0,1))
    elif (args['arg12']=="Entre 100 DHs et 1000 DHs"):
        arr.extend((0,1,0))
    elif (args['arg12']=="Entre 1000 DHs et 100000 DHs"):
        arr.extend((1,0,0))
    else:
        arr.extend((0,0,0))
    if (args['arg13']=="Figures d'autorité (Ex: Professeur, patron, parent ou membre de la famille plus âgée etc)"):  
        arr.extend((1,0,0))
    elif (args['arg13']=="Amicale"):
        arr.extend((0,1,0))
    elif (args['arg13']=="Amoureuse"):
        arr.extend((0,0,1))
    else:
        arr.extend((0,0,0))     
    if (args['arg14']=="Luxurieux"):  
        arr.extend((1,0))
    elif (args['arg14']=="Personnalisé"):
        arr.extend((0,1))
    else:
        arr.extend((0,0))
    print(arr)
    features=np.array(arr)
    featuresReshaped=features.reshape(1,-1)
    prediction=model.predict(featuresReshaped)
    #print(prediction)
    arr.clear()
    return  jsonify({'prediction':prediction[0]})
   


if __name__=="__main__":
    app.run(debug=True)