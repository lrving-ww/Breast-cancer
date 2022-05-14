library(limma)
library(glmnet)
library(survival)
library(survivalROC)
library(pheatmap)
library(ggpubr)
library(ggthemes)
library(ggplot2)
clinical<-read.table(file = "H:/ruxianai/BRCA_clinical2.txt",row.names=1,header=T,sep="\t")
dim(clinical)

genes<-read.csv("H:/ruxianai/HiSeqV2.csv",row.names=1,header=T,sep="\t")
dim(genes)


clinical0<-clinical[,c('OS','OS.time','sample_type')]
dim(clinical0)


index<-intersect(rownames(clinical0),colnames(genes))
genes0<-genes[,index]
clinical2<-clinical0[index,]

genes_clinical<-na.omit(cbind(t(genes0),clinical2))
genes<-t(genes_clinical[,1:20530])
clinical<-genes_clinical[,20531:20533]
dim(genes_clinical)
dim(genes)
dim(clinical)






nor<-subset(clinical,sample_type == "Solid Tissue Normal")
pri<-subset(clinical,sample_type=="Primary Tumor")

index_nor<-intersect(rownames(nor),colnames(genes))
index_pri<-intersect(rownames(pri),colnames(genes))

clinical_nor<-nor[index_nor,]
clinical_pri<-pri[index_pri,]

genes_nor<-genes[,index_nor]
genes_pri<-genes[,index_pri]

#training_name = sample(rownames(clinical_pri),num_training)
training_name = rownames(clinical_pri)
training_clinical = clinical_pri[training_name,]
training_genes = genes_pri[,training_name]

test_name = setdiff(rownames(clinical_pri),rownames(training_clinical))
test_clinical = clinical_pri[test_name,]
test_genes = genes_pri[,test_name]

genesdiff<-cbind(training_genes,genes_nor)
exp_kirc<-as.matrix(genesdiff)
genesdiff






#######Gene differential expression analysis

samps<-factor(c(rep("pri",dim(training_clinical)[1]),rep("nor",dim(clinical_nor)[1])))
design<-model.matrix(~0+samps)
colnames(design)<-c("pri","nor")
design
cont.matrix<-makeContrasts(pri-nor,levels=design)
cont.matrix
fit<-lmFit(exp_kirc,design)
fit2<-contrasts.fit(fit,cont.matrix)
fit2<-eBayes(fit2)
final<-topTable(fit2, coef=1,number=dim(exp_kirc)[1],adjust.method="BH",sort.by="B") 
final<-na.omit(final)
deg<-subset(final, adj.P.Val<0.001&abs(logFC)>0.7&AveExpr>5&B>5) 
write.csv(deg,file='F:/deg.csv')



deg.names <- rownames(deg)
deg.data.pri<-training_genes[deg.names,]
degsort0<-deg[order(deg$logFC,decreasing = T),]
degsort<-exp_kirc[rownames(degsort0),]
deg.v<-final
deg.v$logP<- -log10(deg.v$adj.P.Val)
deg.v$Group<-"not-significant"
deg.v$Group[which((deg.v$adj.P.Val<0.05) & (deg.v$logFC > 0.7) & (deg.v$AveExpr > 5) & (deg.v$B > 5))]<-"up-regulated"
deg.v$Group[which((deg.v$adj.P.Val<0.05) & (deg.v$logFC < -0.7) & (deg.v$AveExpr > 5) & (deg.v$B > 5))]<-"down-regulated"
table(deg.v$Group)
deg.v$symbol <- rownames(deg.v)
deg.v$label <- ""
deg.v <- deg.v[order(deg.v$adj.P.Val),]
up.genes <- head(deg.v$symbol[which(deg.v$Group == "up-regulated")],13)
down.genes <- head(deg.v$symbol[which(deg.v$Group == "down-regulated")],13)
deg.top10.gene <- c(as.character(up.genes),as.character(down.genes))
deg.v$label[match(deg.top10.gene,deg.v$symbol)] <- deg.top10.gene

ggscatter(deg.v,x="logFC",y="logP",
          color="Group",
          palette = c("#2f5688", "#BBBBBB", "#CC0000"),
          size = 2,
          label = deg.v$label,
          font.label = 13,
          repel = T,
          xlab = "log2(FoldChange)",
          ylab = "-log10(Adjust P-value)") + theme_pubr() +
  geom_hline(yintercept = 1.30, linetype = "dashed") +
  geom_vline(xintercept = c(-0.7,0.7),linetype = "dashed")


rfs<-training_clinical$OS.time
rfs.ind<-training_clinical$OS
tdeg.data.pri<-t(deg.data.pri)
deg.data.pri.cl<-cbind(tdeg.data.pri,rfs,rfs.ind)
deg.data.pri.cl<-na.omit(deg.data.pri.cl)
deg.data.pri.cl1<-as.data.frame(deg.data.pri.cl)
time<-deg.data.pri.cl1$rfs
status<-deg.data.pri.cl1$rfs.ind
dim(tdeg.data.pri)







#COX
n<-dim(deg.data.pri)[1]
p<-NULL
hr<-NULL
coef<-NULL
low<-NULL
up<-NULL
for(i in c(1:n)){
  tt<-coxph(Surv(rfs,rfs.ind)~tdeg.data.pri[,i])
  t<-summary(tt)
  p<-c(p,t$coefficients[[5]])
  hr<-c(hr,t$coefficients[[2]])
  coef<-c(coef,t$coefficients[[1]])
  low<-c(low,t$conf.int[[3]])
  up<-c(up,t$conf.int[[4]])
}
unicoxrna<-cbind(colnames(tdeg.data.pri),p,hr,coef,low,up)
unicoxrna0.05<-subset(unicoxrna,p<0.05)#ɸѡ????
rownames(unicoxrna0.05)<-unicoxrna0.05[,1]
uniconxrna0.05<-unicoxrna0.05[,-1]
unigenes<-genes_pri[rownames(uniconxrna0.05),]
candigene<-unigenes[,rownames(deg.data.pri.cl1)]
tcandigene<-t(candigene)
dim(tcandigene)
dim(unicoxrna)
dim(uniconxrna0.05)
unicoxrnay<-unicoxrna[,-1]
write.csv(unicoxrna,file='F:/unicoxrna.csv')
write.csv(uniconxrna0.05,file='F:/uniconxrna0.05.csv')





##############Time of live
index1<-intersect(colnames(genes_pri), rownames(clinical_pri))
genes1<-genes_pri[,index1]
clinical1<-clinical_pri[index1,]
genes1<-t(genes1)
time1<-as.data.frame(clinical1$OS.time)
genes1<-genes1[,colnames(tcandigene)]
names(time1)<-'rs'
time1<-as.data.frame(time1)

rs<-time1$rs>365*5
rs<-as.data.frame(rs)
rs[rs=="TRUE"]='LOW'
rs[rs=="FALSE"]='HIGH'
rs<-as.data.frame(rs)

write.table(genes1,file='H:/ruxianai/genes2.txt',row.names=FALSE,quote=FALSE, sep =",")
write.table(rs,file='H:/ruxianai/rs.csv', row.names=FALSE,quote=FALSE, sep =",")





##############age
index1<-intersect(colnames(genes_pri), rownames(clinical_pri))
genes1<-genes_pri[,index1]
clinical1<-clinical_pri[index1,]
genes1<-t(genes1)
time1<-as.data.frame(clinical1$Age_at_Initial_Pathologic_Diagnosis_nature2012)
genes1<-genes1[,colnames(tcandigene)]
names(time1)<-'rs'
rs<-time1$rs>65
rs<-as.data.frame(rs)
rs[rs=="TRUE"]='LOW'
rs[rs=="FALSE"]='HIGH'
rs<-as.data.frame(rs)
write.table(genes1,file='H:/ruxianai/genes2.txt',row.names=FALSE,quote=FALSE, sep =",")
write.table(rs,file='H:/ruxianai/rs.csv', row.names=FALSE,quote=FALSE, sep =",")






######The stage of the patient
index1<-intersect(colnames(genes_pri), rownames(clinical_pri))
genes1<-genes_pri[,index1]
clinical1<-clinical_pri[index1,]
genes1<-t(genes1)
time1<-as.data.frame(clinical1$pathologic_T)
genes1<-genes1[,colnames(tcandigene)]
names(time1)<-'rs'
rs<-time1
write.table(genes1,file='H:/ruxianai/genes2.txt',row.names=FALSE,quote=FALSE, sep =",")
write.table(rs,file='H:/ruxianai/rs.csv', row.names=FALSE,quote=FALSE, sep =",")








#################################State of the tumor with_tumor   tumor_free
index1<-intersect(colnames(genes_pri), rownames(clinical_pri))
genes1<-genes_pri[,index1]
clinical1<-clinical_pri[index1,]
genes1<-t(genes1)
time1<-as.data.frame(clinical1$person_neoplasm_cancer_status)
genes1<-genes1[,colnames(tcandigene)]
#genes1<-genes1[,rownames(combine)]
names(time1)<-'rs'
rs<-time1
write.table(genes1,file='H:/ruxianai/genes2.txt',row.names=FALSE,quote=FALSE, sep =",")
write.table(rs,file='H:/ruxianai/rs.csv', row.names=FALSE,quote=FALSE, sep =",")






