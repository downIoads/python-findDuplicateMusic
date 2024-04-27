# python-findDuplicateMusic
Python tool that is able to find duplicate music tracks (robust against different versions, length, filetypes, etc).
Feel free to play around with parameters or add other metrics, open pull request if you are able to find sth that finds even less false duplicates than current tuning!

## Read the comments in code pls
Code works pretty well but is really slow, especially if you throw more than 50 audio files at it (will recursively pick up any music file in any subfolder, so be careful)
