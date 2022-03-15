for i in $(seq 1 18)
  do  
    echo -e "starting ${i}"
	  python trainSet.py &
	  sleep 14
  done