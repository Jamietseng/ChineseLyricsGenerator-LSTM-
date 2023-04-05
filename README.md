# ChineseLyricsGenerator-LSTM-
2023Fall 文字探勘初論期末專案

本專案目標為製作出中文流行歌詞生成器，透過訓練一歌詞生成器，匯集前人創作的精華，協助音樂創作者有效地產生流行歌歌詞，也提供更廣大的族群投入創作的機會，作為創作入門的好幫手。

本專案透過資料爬蟲，使用魔鏡歌詞網作為資料庫，大量蒐集中文流行歌的歌詞，前處理篩選出純中文的歌詞、區分歌手，使用Ckip斷詞方法進行Tokenization，並使用N-gram、LSTM, GRU, GPT-2等四種NLP模型訓練資料庫內容，計算中文歌詞生成權重與文字間的相似性，進而製作中文歌詞生成器

使用LSTM模型訓練建立好的資料庫內容，進而生成中文歌詞。

資料庫內容：得過金曲僅最佳男歌手獎之男歌手的中文歌詞，已透過Ckip斷詞法進行Tokenization

模型生成結果：礙於電腦記憶體不足，無法使用所有資料庫內容訓練模型，因此使用資料庫中前1500首歌詞進行訓練，印出結果並存於Excel檔中
