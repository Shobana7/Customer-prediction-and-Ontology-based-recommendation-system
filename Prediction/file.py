from flask import Flask,render_template, request
import runcombine
import newcombine
import pandas as pd  
import IntegrationAtRisk
import IntegrationDefiniteCustomer

app = Flask(__name__)

lstm_model = newcombine.call_for_lstm_train()
rf_model = newcombine.call_for_rf_train()

@app.route('/')
def hello_python():
   return '<h1>Hello Python</h1>'

@app.route('/buttonpage')
def buton_page():
   return render_template('buttonpage.html')

@app.route('/notbuying')
def not_buying():
   return render_template('notbuying.html')

@app.route('/sample',methods = ['POST', 'GET'])
def sample():
   if request.method == 'POST':

      result= request.form['clicked_btn']
      result=result.split('/')
      prodID=result[0]
      prodName=result[1]
      prodPrice=result[2]

      Name,ID,AvgPrice,MinPrice,MaxPrice=runcombine.getsamedata() 
      val_lstm=runcombine.get_lstm_output(lstm_model) 
      print("from file.py lstm value")
      print(val_lstm)
      if(val_lstm==1):              #LSTM is 1 Customer is definitely buying!
         high_price_list=IntegrationDefiniteCustomer.run_recommendation_for_input(ID,prodID)
         print("printing return value of reco")
         print(high_price_list)
         if(high_price_list==1):
            pstmt="Hi! Looks like you're a new user! Give us some information about yourself for better recommendations!"
            return render_template('notbuying.html', pstmt=pstmt, prodPrice=prodPrice, prodName=prodName, prodID=prodID, Name=Name,ID=ID,AvgPrice=AvgPrice, MinPrice=MinPrice, MaxPrice=MaxPrice )
         elif(high_price_list==2):
            pstmt="Hi! Looks like you've found a great product!"
            return render_template('notbuying.html', pstmt=pstmt, prodPrice=prodPrice, prodName=prodName, prodID=prodID, Name=Name,ID=ID,AvgPrice=AvgPrice, MinPrice=MinPrice, MaxPrice=MaxPrice )
         else:
            return render_template('displaysuggestions.html',suggestion_list=high_price_list, prodPrice=prodPrice, prodName=prodName, prodID=prodID, Name=Name,ID=ID,AvgPrice=AvgPrice, MinPrice=MinPrice, MaxPrice=MaxPrice)
         
      else:                         #LSTM is 0 Customer is likely to abandon!
         val_rf=runcombine.get_rf_output(rf_model)
         print("from file.py rf value")
         print(val_rf)
         if(val_rf==1):             #LSTM is 0 RF is 1 Customer has purchasin intent but is gonna leave so -->At-risk Customer
            suggestion_list=IntegrationAtRisk.run_recommendation_for_input(ID,prodID)
            print("printing return value of reco")
            print(suggestion_list)
            if(suggestion_list==1):
               pstmt="Hi! Looks like you're a new user! Give us some information about yourself for better recommendations!"
               return render_template('notbuying.html', pstmt=pstmt,prodPrice=prodPrice, prodName=prodName, prodID=prodID, Name=Name,ID=ID,AvgPrice=AvgPrice, MinPrice=MinPrice, MaxPrice=MaxPrice )
            elif(suggestion_list==2):
               pstmt="Hi! Looks like you've found a great product!"
               return render_template('notbuying.html', pstmt=pstmt,prodPrice=prodPrice, prodName=prodName, prodID=prodID, Name=Name,ID=ID,AvgPrice=AvgPrice, MinPrice=MinPrice, MaxPrice=MaxPrice )
            else:
               return render_template('displaysuggestions.html',suggestion_list=suggestion_list,prodPrice=prodPrice, prodName=prodName, prodID=prodID, Name=Name,ID=ID,AvgPrice=AvgPrice, MinPrice=MinPrice, MaxPrice=MaxPrice)
         else:                      #LSTM is 0 RF is 0 Customer will abandon the site, not intention to purchase
            pstmt='Happy Shopping!'
            return render_template('notbuying.html', pstmt=pstmt, prodPrice=prodPrice, prodName=prodName, prodID=prodID, Name=Name,ID=ID,AvgPrice=AvgPrice, MinPrice=MinPrice, MaxPrice=MaxPrice )
         

@app.route('/chooseprod',methods = ['POST', 'GET'])
def choose_prod():
   if request.method == 'POST':
      Name,ID,AvgPrice,MinPrice,MaxPrice=runcombine.givedata() 
      return render_template('chooseprod.html',Name=Name,ID=ID,AvgPrice=AvgPrice, MinPrice=MinPrice, MaxPrice=MaxPrice)


if __name__ == '__main__':
   app.run(debug = True)
