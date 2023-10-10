
library(data.table)
library(dplyr)
library(ggplot2)
library(caret)
library(corrplot)
library(cowplot)
library(xgboost)

setwd("C:\\Users\\chott\\Downloads\\Projects\\Big Mart Sales Prediction")
train=fread("train_v9rqX0R.csv")
test=fread("test_AbJTz2l.csv")
submission=fread("sample_submission_8RXa3c6.csv")

#Dimensions of the data
dim(train);dim(test)

#Features of the data
names(test);names(test)

#Structure of the data
str(train);str(test)

#Combining training and test
test[,Item_Outlet_Sales:=NA]
df=rbind(train,test)
dim(df)

#Target variable
ggplot(train)+geom_histogram(aes(train$Item_Outlet_Sales),binwidth = 100,fill="slategrey")+xlab("Item Outlet Sales")


#Independent Variables

p1=ggplot(df)+geom_histogram(aes(Item_Weight),binwidth = 0.5,fill="lightblue")
p2=ggplot(df)+geom_histogram(aes(Item_Visibility),binwidth = 0.05,fill="lightblue")
p3=ggplot(df)+geom_histogram(aes(Item_MRP),binwidth = 1,fill="lightblue")
plot_grid(p1,p2,p3,nrow=1)

#Categorical variables
ggplot(df%>%group_by(Item_Fat_Content)%>% summarise(Count=n()))+geom_bar(aes(Item_Fat_Content,Count),stat="identity",fill="coral")

#Converting LF,low fat and reg into Low Fat and Regular
df$Item_Fat_Content[df$Item_Fat_Content=="LF"]="Low Fat"
df$Item_Fat_Content[df$Item_Fat_Content=="low fat"]="Low Fat"
df$Item_Fat_Content[df$Item_Fat_Content=="reg"]="Regular"
ggplot(df%>%group_by(Item_Fat_Content)%>% summarise(Count=n()))+geom_bar(aes(Item_Fat_Content,Count),stat="identity",fill="coral")

#Plotting Item type,Outlet identifier and Outlet size
p4=ggplot(df%>%group_by(Item_Type)%>%summarise(Count=n()))+
  geom_bar(aes(Item_Type,Count),stat="identity",fill="coral")+
  geom_label(aes(Item_Type,Count,label=Count),vjust=0.5)+
  theme(axis.text.x = element_text(angle = 45,hjust = 1))+
  theme(plot.title = element_text(hjust=0.5))+
  xlab("")+
  ggtitle("Item Type")

p5=ggplot(df%>%group_by(Outlet_Identifier)%>%summarise(Count=n()))+
  geom_bar(aes(Outlet_Identifier,Count),stat="identity",fill="coral")+
  geom_label(aes(Outlet_Identifier,Count,label=Count),vjust=0.5)+
  theme(axis.text.x = element_text(angle = 45,hjust = 1))

p6=ggplot(df%>%group_by(Outlet_Size)%>%summarise(Count=n()))+
  geom_bar(aes(Outlet_Size,Count),stat="identity",fill="coral")+
  geom_label(aes(Outlet_Size,Count,label=Count),vjust=0.5)+
  theme(axis.text.x = element_text(angle = 45,hjust = 1))

second_row=plot_grid(p5,p6,nrow = 1)
plot_grid(p4,second_row,ncol=1)


#Plotting outlet establishment and outlet type
p7=ggplot(df%>%group_by(Outlet_Establishment_Year)%>%summarise(Count=n()))+
  geom_bar(aes(factor(Outlet_Establishment_Year),Count),stat="identity",fill="coral")+
  geom_label(aes(factor(Outlet_Establishment_Year),Count,label=Count),vjust=0.5)+
  xlab("Outlet Establishment Year")+
  theme(axis.text.x = element_text(size=8.5))

p8=ggplot(df%>%group_by(Outlet_Type)%>%summarise(Count=n()))+
  geom_bar(aes(Outlet_Type,Count),stat="identity",fill="coral")+
  geom_label(aes(Outlet_Type,Count,label=Count),vjust=0.5)+
  xlab("Outlet Type")
  theme(axis.text.x = element_text(size=8.5))

plot_grid(p7,p8,ncol = 2)

#Target variable vs independent numerical variables
p9=ggplot(train)+
  geom_point(aes(Item_Weight,Item_Outlet_Sales),col="palevioletred",alpha=0.4)+
  theme(axis.title = element_text(size = 8.5))

p10=ggplot(train)+
  geom_point(aes(Item_Visibility,Item_Outlet_Sales),col="palevioletred",alpha=0.4)+
  theme(axis.title = element_text(size = 8.5))

p11=ggplot(train)+
  geom_point(aes(Item_MRP,Item_Outlet_Sales),col="palevioletred",alpha=0.4)+
  theme(axis.title = element_text(size = 8.5))

second_row_2=plot_grid(p10,p11,ncol=2)
plot_grid(p9,second_row_2,nrow=2)

#Target Variable vs Independent categorical variables
p12=ggplot(train)+
  geom_violin(aes(Item_Type,Item_Outlet_Sales),fill="turquoise")+
  theme(axis.text.x = element_text(angle = 45,hjust = 1),
        axis.text = element_text(size = 6),
        axis.title = element_text(size = 8.5))

p13=ggplot(train)+
  geom_violin(aes(Item_Fat_Content,Item_Outlet_Sales),fill="turquoise")+
  theme(axis.text.x = element_text(angle = 45,hjust = 1),
        axis.text = element_text(size = 8),
        axis.title = element_text(size = 8.5))

p14=ggplot(train)+
  geom_violin(aes(Outlet_Identifier,Item_Outlet_Sales),fill="turquoise")+
  theme(axis.text.x = element_text(angle = 45,hjust = 1),
        axis.text = element_text(size = 8),
        axis.title = element_text(size = 8.5))

second_row_3=plot_grid(p13,p14,ncol=2)
plot_grid(p12,second_row_3,nrow = 2)

ggplot(train)+
  geom_violin(aes(Outlet_Size,Item_Outlet_Sales),fill="rosybrown")

p15=ggplot(train)+
  geom_violin(aes(Outlet_Location_Type,Item_Outlet_Sales),fill="olivedrab3")

p16=ggplot(train)+
  geom_violin(aes(Outlet_Type,Item_Outlet_Sales),fill="olivedrab3")

plot_grid(p15,p16,ncol=1)

#Check for missing values in numerical variables
sum(is.na(df$Item_Weight))
sum(is.na(df$Item_Visibility))
sum(is.na(df$Item_MRP))

#imputing missing values 
missing_index=which(is.na(df$Item_Weight))

for (i in missing_index) {
  item=df$Item_Identifier[i]
  df$Item_Weight[i]=mean(df$Item_Weight[df$Item_Identifier==item],na.rm=T)
}
sum(is.na(df$Item_Weight))

#Replacing 0's in Item_Visibility
ggplot(df)+
  geom_histogram(aes(Item_Visibility),bins=100,fill="slategrey")

zero_index=which(df$Item_Visibility==0)

for (i in zero_index) {
  item=df$Item_Identifier[i]
  df$Item_Visibility[i]=mean(df$Item_Visibility[df$Item_Identifier==item],na.rm=T)
}

ggplot(df)+
  geom_histogram(aes(Item_Visibility),bins=100,fill="slategrey")

#Feature Engineering
perishable=c("Breads","Breakfast","Dairy","Fruits and Vegetables","Meat","Seafood")
non_perishable=c("Baking Goods","Canned","Frozen Foods","Hard Drinks","Health and Hygiene","Household","Soft Drinks")

df[,Item_Type_new:=ifelse(Item_Type %in% perishable,"perishable",ifelse(Item_Type %in% non_perishable,"non perishable","not_sure"))]

table(df$Item_Type,substr(df$Item_Identifier,1,2))
df[,Item_category:=substr(df$Item_Identifier,1,2)]

df$Item_Fat_Content[df$Item_category=="NC"]="Non Edible"

df[,Outlet_years:=2013-Outlet_Establishment_Year]
df$Outlet_Establishment_Year=as.factor(df$Outlet_Establishment_Year)

df[,price_per_unit_wt:=Item_MRP/Item_Weight]

df[,Item_MRP_clusters:= ifelse(Item_MRP<69,"1st",
                               ifelse(Item_MRP>=69 & Item_MRP<136,"2nd",
                                      ifelse(Item_MRP>=136 & Item_MRP<203,"3rd","4th")))]


#Label Encoding
df[,Outlet_Size_num:=ifelse(Outlet_Size=="Small",0,
                            ifelse(Outlet_Size=="Medium",1,2))]

df[,Outlet_Location_Type_num:=ifelse(Outlet_Location_Type=="Tier 1",0,
                                     ifelse(Outlet_Location_Type=="Tier 2",1,2))]

#removing categorical variables after label encoding
df[,c("Outlet_Size","Outlet_Location_Type"):=NULL]


#One hot encoding
ohe=dummyVars("~.",data=df[,-c("Item_Identifier","Outlet_Establishment_Year","Item_Type")],fullRank = T)
ohe_df=data.table(predict(ohe,df[,-c("Item_Identifier","Outlet_Establishment_Year","Item_Type")]))

df=cbind(df[,"Item_Identifier"],ohe_df)

#Pre-processing Data

#removing skewness
df[,Item_Visibility:=log(Item_Visibility + 1)]
df[,price_per_unit_wt:=log(price_per_unit_wt + 1)]

#scaling numeric predictors
num_vars=which(sapply(df,is.numeric))
num_vars_names=names(num_vars)
df_numeric=df[,setdiff(num_vars_names,"Item_Outlet_Sales"),with=F]
prep_num=preProcess(df_numeric,method=c("center","scale"))
df_numeric_norm=predict(prep_num,df_numeric)
df[,setdiff(num_vars_names,"Item_Outlet_Sales"):=NULL]
df=cbind(df,df_numeric_norm)

train=df[1:nrow(train)]
test=df[(nrow(train)+1):nrow(df)]
test[,Item_Outlet_Sales:=NULL]

#correlated variables
cor_train=cor(train[,-c("Item_Identifier")])
corrplot(cor_train,method = "pie",type = "lower",tl.cex=0.9)

#Linear Model
reg_model=lm(Item_Outlet_Sales~.,data=train[,-c("Item Identifier")])
#preparing dataframe for submission and writing it in a csv file
submission$Item_Outlet_Sales=predict(reg_model,test[,-c("Item_Identifier")])
write.csv(submission,"Linear_reg_submit.csv",row.names = F)

#Lasso model
my_control=trainControl(method = "cv",number = 5)
Grid=expand.grid(alpha=1,lambda=seq(0.001,0.1,by=0.0002))
lasso_model=train(x=train[,-c("Item_Identifier","Item_Outlet_Sales")],y=train$Item_Outlet_Sales,method="glmnet",trControl=my_control,tuneGrid=Grid)

#Ridge model
my_control=trainControl(method = "cv",number = 5)
Grid=expand.grid(alpha=0,lambda=seq(0.001,0.1,by=0.0002))
ridge_model=train(x=train[,-c("Item_Identifier","Item_Outlet_Sales")],y=train$Item_Outlet_Sales,method="glmnet",trControl=my_control,tuneGrid=Grid)

#Random Forest
my_control=trainControl(method = "cv",number = 5)
tgrid=expand.grid(
  .mtry=c(3:10),
  .splitrule="variance",
  .min.node.size=c(10,15,20)
)
rf_model=train(x=train[,-c("Item_Identifier","Item_Outlet_Sales")],
               y=train$Item_Outlet_Sales,
               method="ranger",
               trControl=my_control,
               tuneGrid=tgrid,
               num.trees=400,
               importance="permutation")

#best model parameters from plot
plot(rf_model)

#variable importance
plot(varImp(rf_model))

#XGboost
param_list=list(objective="reg:linear",eta=0.01,gamma=1,max_depth=6,sub_sample=0.08,colsample_bytree=0.5)
dtrain=xgb.DMatrix(data=as.matrix(train[,-c("Item_Identifier","Item_Outlet_Sales")]),label=train$Item_Outlet_Sales)
dtest=xgb.DMatrix(data=as.matrix(train[,-c("Item_Identifier")]))
set.seed(112)
xgbcv=xgb.cv(params=param_list,data=dtrain,nrounds=1000,nfold=5,print_every_n = 10,early_stopping_rounds = 30,maximize = F)

#training the model
xgb_model=xgb.train(data=dtrain,params = param_list,nrounds = 430)
#variable importance
var_imp=xgb.importance(feature_names = setdiff(names(train),c("Item_Identifier","Item_Outlet_Sales")),model=xgb_model)
xgb.plot.importance(var_imp)
