#!/bin/sh

echo "Which compiler ? : "
echo " 1. PGI "
echo " 2. CAPS "
echo " 3. CRAY "
echo " 4. PGI and CAPS "
echo " 5. PGI and CRAY "
echo " 6. CAPS and CRAY "
echo " 7. PGI, CAPS and CRAY "
read com
if [ -r Makefile.config ]; then cp -f Makefile.config Makefile.config.bak ; fi
cp Makefile.config.org Makefile.config
case $com in
	1)
	mv -f Makefile.config Makefile.config.tmp
 	cat Makefile.config.tmp | sed 's/useCAPS = true/useCAPS = false/' | sed 's/useCRAY = true/useCRAY = false/' > Makefile.config 
 	\rm -rf Makefile.config.tmp
	;;
	2)
	mv -f Makefile.config Makefile.config.tmp
 	cat Makefile.config.tmp | sed 's/usePGI = true/usePGI = false/' | sed 's/useCRAY = true/useCRAY = false/' > Makefile.config 
 	\rm -rf Makefile.config.tmp
	;;
	3)
	mv -f Makefile.config Makefile.config.tmp
 	cat Makefile.config.tmp | sed 's/useCAPS = true/useCAPS = false/' | sed 's/usePGI = true/usePGI = false/' > Makefile.config 
 	\rm -rf Makefile.config.tmp
	;;
	4)
	mv -f Makefile.config Makefile.config.tmp
 	cat Makefile.config.tmp | sed 's/useCRAY = true/useCRAY = false/' > Makefile.config 
 	\rm -rf Makefile.config.tmp
	;;
	5)
	mv -f Makefile.config Makefile.config.tmp
 	cat Makefile.config.tmp | sed 's/useCAPS = true/useCAPS = false/' > Makefile.config 
 	\rm -rf Makefile.config.tmp
	;;
	6)
	mv -f Makefile.config Makefile.config.tmp
 	cat Makefile.config.tmp | sed 's/usePGI = true/usePGI = false/' > Makefile.config 
 	\rm -rf Makefile.config.tmp
	;;
	7)
	;;
esac

if [ $com == 2 -o $com == 4 -o $com == 6 -o $com == 7 ] ; then
echo "Which compiler for CAPS's host ? : "
echo " 1. INTEL "
echo " 2. GNU "
echo " 3. OTHERS "
read host
case $host in
	1)
	mv -f Makefile.config Makefile.config.tmp
 	cat Makefile.config.tmp | sed 's/CAPS_HOST = gfortran/CAPS_HOST = ifort/' > Makefile.config 
 	\rm -rf Makefile.config.tmp
	;;
	2)
	;;
	3)
	echo " please modify CAPS_HOST in Makefile.config "
	;;
esac
fi


if [ $com == 1 -o $com == 4 -o $com == 5 -o $com == 7 ] ; then
echo "Use CUDA ? : (yes/no) "

while [ 1 ]  
do
    read cuda
    case $cuda in
	yes|YES|y|Y)
	    mv -f Makefile.config Makefile.config.tmp
 	    cat Makefile.config.tmp | sed 's/useCUDA = false/useCUDA = true/' > Makefile.config 
 	    \rm -rf Makefile.config.tmp
	    break
	    ;;
	no|NO|n|N)
	    break
	    ;;
	*)
	    echo "Please input yes or no "
	    ;;
	esac
done
fi

echo "Which GPU ? : "
echo " 1. Kepler "
echo " 2. Fermi "
echo " 3. Others "
read gpu
case $gpu in
	1)
	;;
	2)
	mv -f Makefile.config Makefile.config.tmp
 	cat Makefile.config.tmp | sed 's/usedGPU = Kepler/usedGPU = Fermi/' > Makefile.config 
 	\rm -rf Makefile.config.tmp
	;;
	3)
	mv -f Makefile.config Makefile.config.tmp
 	cat Makefile.config.tmp | sed 's/usedGPU = Kepler/usedGPU = Others/' > Makefile.config 
 	\rm -rf Makefile.config.tmp
	;;
esac


echo "Do you have fpp ? : (yes/no) "

while [ 1 ]  
do
    read fpp
    case $fpp in
	yes|YES|y|Y)
	    break
	    ;;
	no|NO|n|N)
	    mv -f Makefile.config Makefile.config.tmp
 	    cat Makefile.config.tmp | sed 's/useFPP = true/useCUDA = false/' > Makefile.config 
 	    \rm -rf Makefile.config.tmp
	    break
	    ;;
	*)
	    echo "Please input yes or no "
	    ;;
	esac
done


echo "---------------------------------------------"
echo " Makefile.config has been prepared "
echo " Type 'make' to compile"
