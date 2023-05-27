1. vanilla (batch ==1 ) -> failed
2. vanilla (batch ==16) -> 49.674 mAP
3. modified (add +1 feature map) -> 46.527 mAP
4. (3) + (change extra's layers' filer's number) -> 47.944 mAP
5. (4) + make all feature map go to L2Norm -> ??? mAP
6. (*) +N feature map ? (change L2Norm + etc ) -> ??? mAP
                      
