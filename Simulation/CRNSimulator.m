(* ::Package:: *)

(* ::Text:: *)
(*Chemical Reaction Network (CRN) Simulator package is developed by David Soloveichik, copyright 2009.*)
(*http://www.dna.caltech.edu/~davids/*)


(* ::Section:: *)
(*Public interface specification*)


BeginPackage["CRNSimulator`", {"Notation`"}];


rxn::usage="Represents an irreversible reaction. eg. rxn[a+b,c,1]";
revrxn::usage="Represents a reversible reaction. eg. revrxn[a+b,c,1,1]";
conc::usage="Initial concentration: conc[x,10] or conc[{x,y},10].";


Notation[\!\(\*
TagBox[
RowBox[{"r_", 
OverscriptBox["\[LongRightArrow]", "k_"], "p_"}],
"NotationTemplateTag"]\) \[DoubleLongLeftArrow] \!\(\*
TagBox[
RowBox[{"rxn", "[", 
RowBox[{"r_", ",", "p_", ",", "k_"}], "]"}],
"NotationTemplateTag"]\)]


SimulateRxnsys::usage=
"SimulateRxnsys[rxnsys,endtime] simulates the reaction system rxnsys for time 0 \
to endtime. In rxnsys, initial concentrations are specified by conc statements. \
Any options specified (eg AccuracyGoal->13,PrecisionGoal->13,WorkingPrecision->15) \
are passed to NDSolve."; 
SpeciesInRxnsys::usage=
"SpeciesInRxnsys[rxnsys] returns the species in reaction system rxnsys. \
SpeciesInRxnsys[rxnsys,pttrn] returns the species in reaction system rxnsys \
matching Mathematica pattern pttrn (eg x[1,_]).";
SpeciesInRxnsysStringPattern::usage=
"SpeciesInRxnsysPattern[rxnsys,pttrn] returns the species in reaction system rxnsys \
matching Mathematica string pattern pttrn. \
(Eg \"g$*\" matches all species names starting with \"g$\" ; \ 
can also do RegularExpression[\"o..d.\$.*\"].)";
RxnsysToOdesys::usage=
"RxnsysToOdesys[rxnsys,t] returns the ODEs corresponding to reaction system rxnsys. \
If rxnsys includes conc statements, the ODEs include initial conditions.\
The time variable is given as the second argument; if omitted it is set to Global`t.
If option InitialConditions->False is given, ignores conc statements and \
leaves initial concentrations unspecified in the produced ODEs.";
InitialConditions::usage="InitialConditions is an option to RxnsysToOdesys.";


(*To use instead of Sequence in functions with Hold attribute but not HoldSequence,
like Module, If, etc*)
Seq:=Sequence 


(* ::Section:: *)
(*Private*)


Begin["`Private`"];


(*We want rxn[a+b,c,1] to be different from rxn[b+a,c,1], so we have to set attribute
HoldAll. But we also want to evaluate if any variables can be evaluated.*)
SetAttributes[{rxn,revrxn}, HoldAll]
rxn[rs_Plus,ps_,k_]:=
 ReleaseHold[ReplacePart[rxn[1,ps,k],1->Hold[Plus]@@List@@Unevaluated[rs]]]/;
 Hold@@Unevaluated[rs] =!= Hold@@List@@Unevaluated[rs]
rxn[rs_,ps_Plus,k_]:=
 ReleaseHold[ReplacePart[rxn[rs,1,k],2->Hold[Plus]@@List@@Unevaluated[ps]]]/;
 Hold@@Unevaluated[ps] =!= Hold@@List@@Unevaluated[ps]
rxn[rs_,ps_,k_]:=
 (With[{rse=rs},rxn[rse,ps,k]])/;Head[rs]=!=Plus&&Unevaluated[rs]=!=rs
rxn[rs_,ps_,k_]:=
 (With[{pse=ps},rxn[rs,pse,k]])/;Head[ps]=!=Plus&&Unevaluated[ps]=!=ps
rxn[rs_,ps_,k_]:=
 (With[{ke=k},rxn[rs,ps,ke]])/;Unevaluated[k]=!=k


revrxn[r_,p_,k1_,k2_]:=Sequence[rxn[r,p,k1],rxn[p,r,k2]]
conc[xs_List,c_]:=(conc[#,c]&/@xs)/.List->Sequence


SimulateRxnsys[rxnsys_,endtime_,opts:OptionsPattern[NDSolve]]:=
Module[{
inputspecs=Cases[rxnsys,conc[x_,c_]:>{x,c}],rsys=DeleteCases[rxnsys,conc[___]]},
SimulateRxnsysInitsList[rsys,inputspecs,endtime,opts]]


SpeciesInRxnsys[rsys_]:=Cases[Cases[rsys,rxn[r_,p_,_]:>Seq[r,p]]/.Times|Plus->Seq,_Symbol|_Symbol[__]]//Union
SpeciesInRxnsys[rsys_,pattern_]:=Cases[SpeciesInRxnsys[rsys],pattern]
SpeciesInRxnsysStringPattern[rsys_,pattern_]:=Select[SpeciesInRxnsys[rsys],StringMatchQ[ToString[#],pattern]&]


Options[RxnsysToOdesys]={InitialConditions->True};

RxnsysToOdesys[rxnsys_,t_Symbol:Global`t,OptionsPattern[]]:=
If[OptionValue[InitialConditions],
 Module[
  {spcs=SpeciesInRxnsys[rxnsys],
  inputspecs=Cases[rxnsys,conc[x_,c_]:>{x,c}],eqs=RxnsysToOdesysNoInits[DeleteCases[rxnsys,conc[___]],t],
  initeqs},
  initeqs=(#[0]==Cases[inputspecs,{#,c0_}:>c0]/.{{c0_}:> c0,{}->0})&/@spcs;
  Join[eqs,initeqs]],
 RxnsysToOdesysNoInits[rxnsys,t]]


RxnsysToOdesysNoInits[rsys_,t_Symbol:Global`t]:=
Module[{spcs=SpeciesInRxnsys[rsys], rrates, spccoeffs,odes},
(*create terms for the rates of each reaction*)
rrates = rsys/.rxn[r_,_,k_]:>(k (r/.{Times[b_,s_]:>s^b,Plus->Times}));
(*for each species, get list of net coefficient for each reaction*)
spccoeffs=Coefficient[rsys/.rxn[r_,p_,_]:>p-r,#]& /@ spcs;
(*create ode for each species*)
odes=MapThread[Function[{spc,coeffs},HoldForm[D[spc,t]]==Total[coeffs*rrates]],{spcs, spccoeffs}];
(*change all species conc to be functions of t*)
odes/.s_/;MemberQ[spcs,s]:>s[t]]//ReleaseHold;


(*Takes initial concentration specification in a different format: {{x,1},{y,2}}.
Unspecified species are set to 0. *)
SimulateRxnsysInitsList[rsys_,inputspecs_, endtime_,opts:OptionsPattern[NDSolve]]:=
Module[{spcs=SpeciesInRxnsys[rsys],odesys=RxnsysToOdesysNoInits[rsys,t], NDSolveEqns},
NDSolveEqns = Join[odesys,(#[0]==Cases[inputspecs,{#,c0_}:>c0]/.{{c0_}:> c0,{}->0})&/@spcs];
Quiet[NDSolve[NDSolveEqns, spcs, {t,0,endtime},opts,MaxSteps->Infinity],{NDSolve::"precw"}][[1]]]


End[];
EndPackage[];
