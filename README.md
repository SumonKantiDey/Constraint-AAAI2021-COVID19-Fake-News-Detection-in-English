# COVID19-Fake-News-Detection-in-English [link](https://competitions.codalab.org/competitions/26655#learn_the_details-overview)


## class identification
- Fake (1): post or tweet provide information cosidered as fake.
- Real (0): post or tweet which provided verified information.
## Preprocessing
```
"👑🦠": "coronavirus","📍":"update","🆕":"update","📌":"update","📢":"update","📣":"update","🔴":"warning",
"👍":"approved","🔰":"driving sign","🙏":"thank you","😷":"wear a mask","❗":"excitement","🌍":"europe africa",
"🌎":"america","🌏":"asia austrila","🌐":"globe","🏫":"school","📺":"television","👏":"hand washing",
"🤔":"thinking","🏠":"stay at home","🏡":"stay at home","🕌":"mosque","📈":"increasing","📉":"decreasing",
"🙌":"hand washing","👐":"hand washing","👦":"boy","❌":"not","🙄":"rolling eye","📏":"keep distance",
"👥":"social distance","💉":"vaccination","\U0001f9ea""vaccination","🧪":"vaccination" ,"💊":" pill",
"🗣️":"speak" ,"🚍":"transportation","🚌":"transportation","💺":"seat" ,"🐍":"snake","🏥":"hospital",
"🚑":"ambulance"  ,"🚨":"emergency","🏪":"convenience store","😞":"disappointed" ,"🎉":"party popper",
"📰":"newspaper","🤱":"breast feeding","👀":"astonished","🧤":"gloves","🚶🏿":"social distance","🚥":"traffic light",
"↗":"up right","⚕":"medicine and health care services","↔️":"keep distance","🦠":"virus","💧":"droplet",
"💦":"droplet","🧴":"sanitizer","🧼":"hand wash","🚫":"not","🚰":"hand wash","🤝":"not handshake",
"🧑‍🤝‍🧑":"not close contact","🧫":"test for covid-19","🧪":"test for covid-19","🌡️":"body temperature measured",
"🔬":"test for covid-19","📝":"contact tracing","🧑‍⚕️":"health worker","🐦":"hand wash","🛡":"shielding required",
"♿️":"accessibility requirements","🆘":"emergency help required"
```

## Training and Test files prepare
- **train_data.csv** is preprocessed file from actual Constraint_English_Train.csv
- **test_data.csv** is preprocessed file from actual Constraint_English_Val.csv

## Wordcloud for train and test data
<p align="center">
 <img src="https://github.com/SumonKantiDey/Constraint-AAAI2021-COVID19-Fake-News-Detection-in-English/blob/main/wordcloud.png" >
</p>


## Training Procedure
- Run python **train.py** file from src directory
- See the all train and validation history from **logs/info.log**
- See train and validation history visualization from **train_loss_his/acc_curves.png** and **train_loss_his/loss_curves.png**

## Contribution
- We have done regular preprocessing to replace emojis with tokens , replace all of  the mentions with Twitter handles, split all of the hashtags into separate words. [[preprocessing]](https://github.com/SumonKantiDey/Constraint-AAAI2021-COVID19-Fake-News-Detection-in-English/blob/main/robust%20preprocessing.ipynb)
- Extracted news headlines from URL which was used in user posts and concatenate that with user post if post and headline not syntactically similar. 
- Used transformers-based variant of Bert and Roberta models alongside different architecture on top of Bert and Roberta models.
- Utilized Ensemble-based approach based on top three model.


### To run all of the bert-base-uncased model 
- Enter bert-base-uncased folder[I have shared a bert base pretrain model folder use that as a pretrain path]
- ```chmod +x run.sh``` for permission
- ```./run.sh``` put that command from terminal to run all of the bert base uncased model

### To run all of the bert-large-uncased model 
- Enter bert-large-uncased folder[I have shared a bert large pretrain model folder use that as a pretrain path]
- ```chmod +x run.sh``` for permission
- ```./run.sh``` put that command from terminal to run all of the bert large uncased model

### To run all of the covid bert model 
- Enter covid-twitter-bert folder
- ```chmod +x run.py``` for permission
- ```./run.sh``` put that command from terminal to run all of the covid model

### To run all of the roberta large model 
- Enter roberta base folder
- ```chmod +x run.sh``` for permission
- ```./run.sh``` put that command from terminal to run all of the roberta large model

```
our best model achieved the weighted F1-score of 0.978 on the test
set (26th place in the leaderboard) out of 166 submitted teams in total.
```
