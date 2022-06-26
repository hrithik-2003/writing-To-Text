#include<iostream.h>
#include<conio.h>
#include<fstream.h>
#include<stdio.h>
#include<dos.h>
#include<string.h>
#include<process.h>

void animation()
{

	for(int j=0;j<40;j++)
	{
		cout<<"\n\n\n\n\n\n\n\n\n\n\n\n";
		for(int i=0;i<=j;i++)
		{
		   cout<<" ";
		}
		cout<<"CBSE PROJECT(2019) -> HOTEL MANAGEMENT";
		sound(800 + (5*j));
		delay(40);
		clrscr();
	}
	nosound();
}

//---------------------------------------------------------------------------

void sound()
{
	sound(500);
	delay(100);
	nosound();
}
//---------------------------------------------------------------------------
class hotel
{
	int room_no;
	char name[30];
	char address[50];
	char phone[10];

	public:
	void main_menu();	//to dispay the main menu
	void addlodge();         //for 1nightlodge
	void addeco();           //for economic room
	void addprem();          //for premium room
	void add_room();	//to book a room
	void display();	//to display the customer record
	void rooms();	//to display alloted rooms
	void edit();	//to edit the customer record
	int check(int);	//to check room status
	void modify(int);	//to modify the record
	void delete_rec(int);	//to delete the record
	int ret_room(){return room_no;}
	char* ret_name(){return name;}
	char* ret_address(){return address;}
	char* ret_phone(){return phone;}

	void Enter_Details()
	{
		cout<<" Name: ";
		gets(name);
		sound();
		cout<<" Address: ";
		gets(address);
		sound();
		cout<<" Phone No: ";
		gets(phone);
	}
	void set_room(int r)
	{
		room_no=r;
	}

}h;

//----------------------------------------------------------------------------------
void hotel::main_menu()
{
	int choice;
	char ans;

	do
	{
		clrscr();
		cout<<"\n\t\t\t\t*************";
		cout<<"\n\t\t\t\t* MAIN MENU *";
		cout<<"\n\t\t\t\t*************";
		cout<<"\n\n\n\t\t\t1.Book A Room";
		cout<<"\n\t\t\t2.Customer Record";
		cout<<"\n\t\t\t3.Rooms Allotted";
		cout<<"\n\t\t\t4.Edit Record";
		cout<<"\n\t\t\t5.Exit";
		cout<<"\n\n\t\t\tEnter Your Choice: ";
		cin>>choice;
		sound();

		switch(choice)
		{
			case 1: add_room();
					break;
			case 2: display();
					break;
			case 3: rooms();
					break;
			case 4: edit();
					break;
			case 5: exit(0);break;
			default:
					{
						cout<<"\n\n\t\t\tWrong choice!!!";
						cout<<"\n\t\t\tPress any key to continue!!";
						getch();
						sound();
					}
		}
		cout<<"\n\n\tSO, ANYTHING ELSE TO BE DONE ?(y/n)  ";
		cin>>ans;
	}while((ans=='y')||(ans=='Y'));
}
//---------------------------------------------------------------------------

void hotel::addlodge()
{
	clrscr();
	int r,flag;


	cout<<"\n Enter Customer Detalis";
	cout<<"\n **********************";

	while(r>10||r<0)
	{
		cout<<"\n\n(ALERT ! THERE ARE ONLY 10 ROOMS!!)";
		cout<<"\n\n Room no: ";
		cin>>r;
		sound();
	}

	flag=check(r);
	if(flag)
		cout<<"\n Sorry..!!!Room is already booked";
	else
	{       ofstream fout("Record.dat",ios::app|ios::binary);
		h.Enter_Details();
		h.set_room(r);
		sound();
		fout.write((char*)&h,sizeof(hotel));
		cout<<"\nOne night lodge is booked!!!";
		fout.close();

	}

	cout<<"\n Press any key to continue!!";
	getch();
	sound();

}
//--------------------------------------------------------------------------
void hotel::addprem()
{
	clrscr();
	int r,flag;

	cout<<"\n Enter Customer Detalis";
	cout<<"\n **********************";

	while(r>60||r<41)
	{
		cout<<"\n\nALERT ! THERE ARE ONLY 20 PREMIUM ROOMS!!(41 - 60)";
		cout<<"\n\n Room no: ";
		cin>>r;
		sound();
	}

	flag=check(r);
	if(flag)
		cout<<"\n Sorry..!!!Room is already booked";
	else
	{
		ofstream fout("Record.dat",ios::app|ios::binary);
		h.Enter_Details();
		h.set_room(r);
		sound();
		fout.write((char*)&h,sizeof(hotel));
		cout<<"\nPremium room is booked!!!";
		fout.close();

	}

	cout<<"\n Press any key to continue!!";
	getch();
	sound();

}
//--------------------------------------------------------------------------

void hotel::addeco()
{
	clrscr();
	int r,flag;

	cout<<"\n Enter Customer Detalis";
	cout<<"\n **********************";

	while(r>40||r<11)
	{
		cout<<"\n\nALERT ! THERE ARE ONLY 30 economic ROOMS!!(11 - 40)";
		cout<<"\n\n Room no: ";
		cin>>r;
		sound();
	}

	flag=check(r);

	if(flag)
		cout<<"\n Sorry..!!!Room is already booked";
	else
	{
		ofstream fout("Record.dat",ios::app|ios::binary);
		h.Enter_Details();
		h.set_room(r);
		sound();
		fout.write((char*)&h,sizeof(hotel));
		cout<<"\nEconomic room is booked!!!";
		fout.close();

	}


	cout<<"\n Press any key to continue!!";
	getch();
	sound();

}

//--------------------------------------------------------------------------
void hotel::add_room()
{
	clrscr();
	cout<<"\n\n\t\tWHICH TYPE OF ROOM IS NEEDED?\n\n";
	cout<<"\t\t 1 : 1 night lodge\n\t\t 2 : economic room\n\t\t 3 : premium room\n\n";
	int num;
	cout<<"\t\tENTER YOUR CHOICE OF STAY :   ";
	cin>>num;

	switch(num)
	{
		case 1: addlodge();break;
		case 2: addeco();break;
		case 3: addprem();break;
		default : cout<<"\n\n\tWRONG CHOICE ....";
	}


}
//---------------------------------------------------------------------------
void hotel::display()
{
	clrscr();
	ifstream fin("Record.dat",ios::in|ios::binary);
	int r,flag=0;
	cout<<"\n Enter room no: ";
	cin>>r;
	sound();

	while(!fin.eof())
	{
		fin.read((char*)&h,sizeof(hotel));
		if(h.ret_room()==r)
		{
			clrscr();
			cout<<"\n Cusromer Details";
			cout<<"\n ****************";
			cout<<"\n\n Room no: "<<h.ret_room();
			cout<<"\n Name: "<<h.ret_name();
			cout<<"\n Address: "<<h.ret_address();
			cout<<"\n Phone no: "<<h.ret_phone();
			flag=1;
			break;
		}
	}

	if(flag==0)
		cout<<"\n Sorry Room no. not found or vacant!!";

	cout<<"\n\n Press any key to continue!!";
	getch();
	sound();
	fin.close();
}
//--------------------------------------------------------------------------
void hotel::rooms()
{
	int f;
	char set[3];
	clrscr();
	cout<<"\n\n\t\tNO OF AVAILABLE ROOMS  :  \n\n";
	ifstream fin("Record.dat",ios::in|ios::binary);
	cout<<"\nROOMS 1 - 10 (low):\n";
	for(int i=1;i<11;i++)
	{
		f=check(i);
		if(f==1)
		{
			strcpy(set,"YES");
			cout<<"  "<<i<<'-'<<set;
		}
		else
		{
			strcpy(set,"NO");
			cout<<"  "<<i<<'-'<<set;
		}
	}
	cout<<"\n\n";
	cout<<"\nROOMS 11 - 40 (econimic):\n";
	for(i=11;i<41;i++)
	{
		f=check(i);
		if(f==1)
		strcpy(set,"YES");
		else
		strcpy(set,"NO");
		cout<<"  "<<i<<'-'<<set;
		if(i%10==0)
		cout<<"\n";
	}

	cout<<"\nROOMS 41 - 60 (premium):\n";
	for(i=41;i<61;i++)
	{
		f=check(i);
		if(f)
		strcpy(set,"YES");

		else
		strcpy(set,"NO");
		cout<<"  "<<i<<'-'<<set;
		if(i%10==0)
		cout<<"\n";
	}
	cout<<"\n\t\t\t ->   YES - BOOKED";
	cout<<"\n\t\t\t ->   NO - NOT BOOKED";
}
//-------------------------------------------------------------------------
void hotel::edit()
{
	clrscr();
	int choice,r;

	cout<<"\n EDIT MENU";
	cout<<"\n *********";
	cout<<"\n\n 1.Modify Customer Record";
	cout<<"\n 2.Delete Customer Record";

	cout<<"\n Enter your choice: ";
	cin>>choice;
	sound();

	clrscr();
	cout<<"\n Enter room no: ";
	cin>>r;

	switch(choice)
	{
		case 1: modify(r);
				break;
		case 2: delete_rec(r);
				break;
		default: cout<<"\n Wrong Choice!!";
	}

	cout<<"\n Press any key to continue!!!";
	getch();
	sound();
}
//--------------------------------------------------------------------------
int hotel::check(int r)
{
	int flag=0;
	ifstream fin("Record.dat",ios::in|ios::binary);
	while(!fin.eof())
	{
		fin.read((char*)&h,sizeof(hotel));
		if(h.ret_room()==r)
		{
			flag=1;
			break;
		}
	}
	fin.close();
	return(flag);
}
//--------------------------------------------------------------------------
void hotel::modify(int r)
{
	long pos,flag=0;
	fstream file("Record.dat",ios::in|ios::out|ios::binary);

	while(!file.eof())
	{

		file.read((char*)&h,sizeof(hotel));
		pos=file.tellg();
		if(h.ret_room()==r)
		{
			cout<<"\n Enter New Details";
			cout<<"\n *****************\n\n";
			h.Enter_Details();

			file.seekp(pos-sizeof(h));
			file.write((char*)&h,sizeof(hotel));
			cout<<"\n Record is modified!!";
			flag=1;
			break;
		}
	}

	if(flag==0)
		cout<<"\n Sorry Room no. not found or vacant!!";

	file.close();
}
//----------------------------------------------------------------------------
void hotel::delete_rec(int r)
{
	int flag=0;
	char ch;
	ifstream fin("Record.dat",ios::in|ios::binary);
	ofstream fout("temp.dat",ios::out|ios::binary);

	while(fin.read((char*)&h,sizeof(hotel)))
	{

		if(h.ret_room()==r)
		{
			cout<<"\n Name: "<<h.ret_name();
			cout<<"\n Address: "<<h.ret_address();
			cout<<"\n Phone No: "<<h.ret_phone();
			cout<<"\n\n Do you want to delete this record(y/n): ";
			cin>>ch;
			sound();

			if(ch=='n')
			fout.write((char*)&h,sizeof(hotel));
			flag=1;
		}
		else
			fout.write((char*)&h,sizeof(hotel));
	}

	fin.close();
	fout.close();

	if(flag==0)
		cout<<"\n Sorry room no. not found or vacant!!";
	else
	{
		remove("Record.dat");
		rename("temp.dat","Record.dat");
	}
}
//--------------------------------------------------------------------------
void main()
{
	clrscr();
	hotel h;
	//textmode(C80);
	textbackground(WHITE);
	textcolor(BLACK);

	animation();

	clrscr();
	cout<<"\n\t\t\t****************************";
	cout<<"\n\t\t\t* HOTEL MANAGEMENT PROJECT *";
	cout<<"\n\t\t\t****************************";
	sleep(1);

	cout<<"\n\n\n\n\t\tMade By:";
	sleep(1);
	cout<<"B.HRITHIK and HRISHIKESH.M";
	sleep(1);
	cout<<"\n\n\n\n\t\tSUBMITTED TO:";
	sleep(1);
	cout<<" Mrs. RACHANA";
	cout<<"\n\n\n\n\n\t\t\t\tPress any key to continue!!";
	getch();
	sound();
	h.main_menu();
}
//---------------------------------------------------------------------------