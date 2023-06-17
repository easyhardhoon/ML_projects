1. vanilla (batch ==1 ) -> failed
2. vanilla (batch ==16) -> 49.674 mAP
3. modified (add +1 feature map) -> 46.527 mAP
4. (3) + (change extra's layers' filer's number) -> 47.944 mAP
5. (4) + make all feature map go to L2Norm ->  failed. too much normalization 
6. (3) + (change extra's layers' filer's number more) + (1,2,3 featuremap via L2Norm) -> failed ;5
7. (6) - (3's feature map via L2Norm_2)  -> 61.677 mAP
8. (7) + (tuning backbone parameter)+tuning nms&detect+tuning bbox_aspect_num+aspect_R)-> 54.782 mAP
9. (8) + (tuning nms*detct) + (extra's other layer filter num) -> 54.752 mAP                

10. (7) / modify layer[:] -> ??? mAP                
11. (9) / modify_layer[:] -> ??? mAP                
12. (11) + (8,9,10,11) featuremap-> ??? mAP                
13. (10) + (8,9,10,11) featuremap + @ -> ??? mAP                

++ optimizer (use ADAGRAD .. ADAM .. Momentum.... )
++ activation func (leaky_Relu)
++ etc (methods for forbiden overfitting)

...etc methods.... [Need update] 
 

