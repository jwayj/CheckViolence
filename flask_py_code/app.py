from flask import Flask, render_template, request
import main
import pandas
import csv

app = Flask(__name__)

@app.route('/')
def file_upload():
    return render_template('index.html')

@app.route('/eda',methods = ['POST', 'GET'])
def eda():
   if request.method == 'POST':
      file = request.files['파일'].read().decode('utf-8') # 파일 불러오기
      print('-'*50)
      print('-'*10,'00. FILE READ : OK /', 'length :',len(file) )
      print('-'*50)

      df1 = main.def_01_txt_to_csv(file)
      print('-'*50)
      print('-'*10,'01. TXT TO CSV : OK /', df1.shape)
      print('-'*50)

      df2 = main.def_02_add_columns(df1)  
      print('-'*50)
      print('-'*10,'02. ADD Columns : OK /', df2.shape)
      print('-'*50)      
      
      report_dict1 = main.def_03_eda(df2)
      print('-'*50)
      print('-'*10,'03. EDA : OK /',report_dict1.keys())
      print('-'*50) 
      for i,j in report_dict1.items():
        print(i,':',j)
      
      return render_template("eda.html",report = report_dict1) 
 


if __name__ == '__main__':
    app.run(debug = True)