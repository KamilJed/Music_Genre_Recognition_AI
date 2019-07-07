# Music_Genre_Recognition_AI
Recognises song slices genre with ~75% accuracy.
Whole songs genres are recognised a liitle bit more accuratly, because we look at multiple slices, not just one.
Seems to be working quite well.

# How to use
1. Create folder tree in project root:  
-data //folder in which the training data will be stored  
-- //inside data create folders with named by genre ie. rock, blues etc.  
-spectrograms //folder in which the generated spectrograms will be stored  
-- //inside spectrograms create folders with named by genre ie. rock, blues etc. as you did with the data folder  
--test //folder in which the spectrograms of test songs will be generated  
-songs //in that folder paste any songs that you want to categorise by trained model  
  
2. Create spectrograms by using spectrogramsCreator.createSpectrograms()  
3. Train model using main.py  
4. Uncomment lines 34-36 in test.py and use it to categorise your songs.
