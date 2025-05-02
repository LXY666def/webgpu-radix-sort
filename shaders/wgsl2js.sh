echo "const upsweep_comp_wgsl = \`" > ./upsweep_comp_wgsl.js
cat ./upsweep_comp.wgsl >> ./upsweep_comp_wgsl.js
echo "\`;" >> ./upsweep_comp_wgsl.js
echo "export default upsweep_comp_wgsl;" >> ./upsweep_comp_wgsl.js

echo "const spine_comp_wgsl = \`" > ./spine_comp_wgsl.js
cat ./spine_comp.wgsl >> ./spine_comp_wgsl.js
echo "\`;" >> ./spine_comp_wgsl.js
echo "export default spine_comp_wgsl;" >> ./spine_comp_wgsl.js

echo "const downsweep_comp_wgsl = \`" > ./downsweep_comp_wgsl.js
cat ./downsweep_comp.wgsl >> ./downsweep_comp_wgsl.js
echo "\`;" >> ./downsweep_comp_wgsl.js
echo "export default downsweep_comp_wgsl;" >> ./downsweep_comp_wgsl.js

echo "const checker_comp_wgsl = \`" > ./checker_comp_wgsl.js
cat ./checker_comp.wgsl >> ./checker_comp_wgsl.js
echo "\`;" >> ./checker_comp_wgsl.js
echo "export default checker_comp_wgsl;" >> ./checker_comp_wgsl.js