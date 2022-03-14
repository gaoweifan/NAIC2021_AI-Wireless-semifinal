
for i in $(seq 1 24)
  do  
    echo -e "starting ${i}"
	python trainSet.py &
	sleep 15
  done
