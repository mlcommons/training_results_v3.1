lines=$(cat $1)
for line in lines
do
ssh $line sudo /home/files/sprbins/spr_drop_caches_3
sudo /home/files/sprbins/spr_compact_memory_1
sudo /home/files/sprbins/spr_numa_balancing 0
sudo /home/files/sprbins/spr_transparent_hugepage_defrag_always
sudo /home/files/sprbins/spr_transparent_hugepage_defrag_never
sudo /home/files/sprbins/spr_transparent_hugepage_never

done
