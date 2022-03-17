for i in $(seq 1 6)
  do  
    echo -e "starting ${i}"
	  nohup python trainSet.py &
	  sleep 14
  done