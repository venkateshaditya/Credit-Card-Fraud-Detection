setwd("C:\\Users\\Aditya\\PycharmProjects\\Data Mining\\Credit Card Fraud Detection")
library(ggplot2)
library(dplyr)
library(corrplot)
data<-read.csv("creditcard.csv")
data[1:1000,]$Class=1

#
for(col in colnames(data)){
  p1<-ggplot(subset(data,Class==1),aes_string(col,fill="Class"))+
  geom_histogram(aes(y=..count../sum(..count..)),alpha=0.6,fill = "#55BB55")+
    ylab("Count")+
    theme(plot.title = element_text(face="bold",hjust = 0.5),strip.text.x = element_text(size = 10,face="bold"),axis.title=element_text(size=14,face="bold"),legend.position="none")
    
  
  
  p2<-ggplot(subset(data,Class==0),aes_string(col,fill="Class"))+
    geom_histogram(aes(y=..count../sum(..count..)),alpha=0.6,fill = "#6666FF")+
  ylab("Count")+
  theme(plot.title = element_text(face="bold",hjust = 0.5),strip.text.x = element_text(size = 10,face="bold"),axis.title=element_text(size=14,face="bold"),legend.position="none")
  
  
  grid.arrange(p1, p2, ncol = 1)
  g<-arrangeGrob(p1, p2, ncol = 1)
  ggsave(filename = paste(col,".png",sep = ""),g)
}

data$Class1<-as.factor(data$Class)
#Time vs Amount graph

ggplot(data %>% arrange(Class),aes(x=Time,y=Amount,colour=Class1))+
  geom_point(alpha=1)+
  scale_colour_manual(values=c("0"="#333399", "1"="#FF0000"))+
  ggtitle("Time VS Amount")+
  ylab("Amount")+
  labs(colour="Class")+
  theme(plot.title = element_text(face="bold",hjust = 0.5),strip.text.x = element_text(size = 10,face="bold"),axis.title=element_text(size=14,face="bold"))


ggsave(filename = paste("TimeVSAmount.png",sep = ""))

#Correlation Plot
# indx <- sapply(d1, is.factor)
# d1[indx] <- lapply(d1[indx], function(x) as.numeric((x)))
res <- cor(data, method = "pearson", use = "complete.obs")
corrplot(res, type = "lower", order = "hclust", 
         tl.col = "black", tl.srt = 6) +
  theme(strip.text = element_text(size = 20, face = "bold"))+ title("Correlation Plot")

#Class Imbalance plot
ggplot(data,aes(Class,fill=Class1))+
  geom_bar(aes(y=..count..),alpha=0.6)+
  ggtitle("Unbalanced Representation")+
  ylab("Count")+
  labs(fill="Class")+
  theme(plot.title = element_text(face="bold",hjust = 0.5),strip.text.x = element_text(size = 10,face="bold"),axis.title=element_text(size=14,face="bold"))
ggsave(filename = paste("ClassImbalance.png",sep = ""))


require(reshape2)
df.m <- melt(data, id.var = "Class")
ggplot(data = df.m, aes(x=variable, y=value)) + geom_boxplot(aes(fill=Class))
