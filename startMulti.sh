for i in $(seq 1 30)
  do  
    echo -e "starting ${i}"
	  python trainSet.py &
	  sleep 13
  done