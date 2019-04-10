cd $TRAVIS_BUILD_DIR/paper
cp ms.tex ms_hogg.tex
sed -i 's/twocolumn/modern/g' ms_hogg.tex
