
Setup script

For ubuntu

``` 
sh bin/setup_ubuntu.sh
```

For mac 

``` 
sh bin/setup_mac.sh
```


Make request using curl 
```
curl -X POST http://127.0.0.1:5000/question_chatbot   -H "X-API-KEY:fpv74NMceEzy.5OsNsX43uhfa2GSGPPOB1/o2ABXg0mMwniAef02" -H "Content-Type: application/json" -d '{"user-input": "Hi"}'
```
