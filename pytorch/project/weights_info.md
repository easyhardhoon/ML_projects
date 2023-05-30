1. vanilla (batch ==1 ) -> failed
2. vanilla (batch ==16) -> 49.674 mAP
3. modified (add +1 feature map) -> 46.527 mAP
4. (3) + (change extra's layers' filer's number) -> 47.944 mAP
5. (4) + make all feature map go to L2Norm ->  failed. too much normalization 
6. (3) + (change extra's layers' filer's number more) + (1,2,3 featuremap via L2Norm) -> failed ;5
7. (6) - (3's feature map via L2Norm_2)  -> ??? mAP
8.  -> ??? mAP
                     

++ optimizer (use ADAGRAD .. ADAM .. Momentum.... )
++ activation func (leaky_Relu)
++ etc (methods for forbiden overfitting) 
