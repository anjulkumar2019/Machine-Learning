
class variable_selection:
    def selectkbest(self, data, predictor):
        x=data.drop(predictor, axis=1)
        y=data[predictor]
        model = LogisticRegression()
        rfe=RFE(model )
        fit = rfe.fit(x,y)
        print(fit.n_features_)
        print(fit.support_)
        print(fit.ranking_)
        print(x.columns)

    def extratreec(self, data, predictor):
        x=data.drop(predictor, axis=1)
        y=data[predictor]
        model=ExtraTreesClassifier()
        model.fit(x,y)
        print(model.feature_importances_)
        
    
