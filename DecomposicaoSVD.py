'''Decomposicao SVD'''
'''Parte 2: Joao Pedro Gomes Araujo Munhoz Ribeiro'''

import numpy as np
import time

#Definicao da matriz devidamente formatada
def Matriz(A):
    M = A@(np.transpose(A))
    return M

#Decompoe as matrizes Q e R
def Matrizes_Q_R(M):
    Q, R = np.linalg.qr(M)
    return Q, R

#Calculo dos valores singulares
def Singulares_QR(B):

   B = np.array(B)
   tol = 1e-15
   W = B.copy()

   for k in range(1000):
      Qk, Rk = Matrizes_Q_R(W)
      Wr = W.copy()
      W = Rk @ Qk
      er = np.linalg.norm(np.diag(W) - np.diag(Wr)) / np.linalg.norm(np.diag(W)) 
      er = np.abs(er)

      if er < tol:
         break

   autoval = []
   for j in range(W.shape[0]):
      autoval.append(np.max(np.sqrt(W[j][j].real),0))

   autoval.sort(reverse= True)
   autoval = np.array(autoval)
   return autoval

         
def Autovetores_Direita(A, autv):
   tol=1e-15
   iter= 1000
   A = np.array(A)
   n= A.shape[0]
   I = np.eye(n)
   autovetores = []

   e = 1e-15

   
   for k in range(n):
      qold = np.random.rand(n)
      mu = autv[k] + e
      B= (A - mu*I)

      norm = 1                                       
      cont = 0                                       
      while norm > tol and cont < iter:              
         qnew = np.linalg.solve(B,qold)              
         qnew = qnew/np.linalg.norm(qnew,ord=2)      
         norm = 1 - np.abs(np.vdot(qold, qnew))      
         qold = qnew                                 
         cont += 1

         autovetores.append(qold)     

   
   autovetores = np.array(autovetores)

   return autovetores

def Autovetores_esquerda(A, autovec, autoval):
   u = []
   n = len(autoval)
   for i in range(n):
      u.append((A @ autovec[i])/autoval[i])

   u = np.array(u)
   return u

def Matrizes(A, AutovecL, AutovecR, Autoval):
   n = A.shape[0]

   U = []
   S = np.eye(n)
   V = []

   for j in range(n):
        U.append(AutovecL[j])
        S[j][j] = Autoval[j]
        V.append(AutovecR[j])

   U = np.array(U)
   V = np.array(V).T

   return U, S, V

def DecompsicaoSVD(A):
   inicio = time.perf_counter()
   M = Matriz(A)
   ValoresSingulares = Singulares_QR(M)
   VecR = Autovetores_Direita(M, ValoresSingulares)
   VecL = Autovetores_esquerda(M,VecR, ValoresSingulares)
   U, S, V = Matrizes(M, VecL, VecR, ValoresSingulares)
   fim = time.perf_counter()
   tempo = fim - inicio
   return U, S, V, tempo

# Comparacao com o svd nativo:

A = np.array([[1,2,3],[9,0,7],[2,5,0]])

u,s,v,t= DecompsicaoSVD(A)
print(f"Minha solução eh: U: {u}\n S: {s}\n V: {v}")
print("O tempo de execucao foi de:", t)


iniProg = time.perf_counter()
m, n, p= np.linalg.svd(A)
fimProg = time.perf_counter()
tempoProg= fimProg - iniProg

print(f"A solucao do programa foi: U: {m}\n S: {n}\n V: {p}")
print("O tempo de execucao do programa foi:", tempoProg)

print(f"A diferenca foi: U: {u - m}\n S: {s - (n @ np.eye(3))}\n Vt: {v - p}")