for i in $(seq 1 12)
  do  
    echo -e "starting ${i}"
	  nohup python trainSet.py &
	  sleep 15
  done