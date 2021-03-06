# COVID19-Fake-News-Detection-in-English [link](https://competitions.codalab.org/competitions/26655#learn_the_details-overview)


## class identification
- Fake (1): post or tweet provide information cosidered as fake.
- Real (0): post or tweet which provided verified information.
## Preprocessing
```
"ππ¦ ": "coronavirus","π":"update","π":"update","π":"update","π’":"update","π£":"update","π΄":"warning",
"π":"approved","π°":"driving sign","π":"thank you","π·":"wear a mask","β":"excitement","π":"europe africa",
"π":"america","π":"asia austrila","π":"globe","π«":"school","πΊ":"television","π":"hand washing",
"π€":"thinking","π ":"stay at home","π‘":"stay at home","π":"mosque","π":"increasing","π":"decreasing",
"π":"hand washing","π":"hand washing","π¦":"boy","β":"not","π":"rolling eye","π":"keep distance",
"π₯":"social distance","π":"vaccination","\U0001f9ea""vaccination","π§ͺ":"vaccination" ,"π":" pill",
"π£οΈ":"speak" ,"π":"transportation","π":"transportation","πΊ":"seat" ,"π":"snake","π₯":"hospital",
"π":"ambulance"  ,"π¨":"emergency","πͺ":"convenience store","π":"disappointed" ,"π":"party popper",
"π°":"newspaper","π€±":"breast feeding","π":"astonished","π§€":"gloves","πΆπΏ":"social distance","π₯":"traffic light",
"β":"up right","β":"medicine and health care services","βοΈ":"keep distance","π¦ ":"virus","π§":"droplet",
"π¦":"droplet","π§΄":"sanitizer","π§Ό":"hand wash","π«":"not","π°":"hand wash","π€":"not handshake",
"π§βπ€βπ§":"not close contact","π§«":"test for covid-19","π§ͺ":"test for covid-19","π‘οΈ":"body temperature measured",
"π¬":"test for covid-19","π":"contact tracing","π§ββοΈ":"health worker","π¦":"hand wash","π‘":"shielding required",
"βΏοΈ":"accessibility requirements","π":"emergency help required"
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
