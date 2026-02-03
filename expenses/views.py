from django.shortcuts import render, redirect, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from .models import Category, Expense, ExpenseLimit
from django.contrib import messages
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.http import JsonResponse
from userpreferences.models import UserPreference
import datetime
from datetime import date
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from django.core.mail import send_mail
from django.conf import settings

# -------------------------------
# NLP / ML Functions
# -------------------------------

def preprocess_text(text):
    """Tokenizes and cleans a text string"""
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    try:
        tokens = word_tokenize(text.lower())
    except LookupError:
        nltk.download('punkt')
        tokens = word_tokenize(text.lower())

    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)


def train_model():
    """Trains a RandomForestClassifier from CSV dataset"""
    data = pd.read_csv('dataset.csv')
    data['clean_description'] = data['description'].apply(preprocess_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['clean_description'])
    model = RandomForestClassifier()
    model.fit(X, data['category'])

    return model, vectorizer


def predict_category_from_description(description):
    """Predicts category using the trained model"""
    model, vectorizer = train_model()
    clean_desc = preprocess_text(description)
    X_input = vectorizer.transform([clean_desc])
    predicted_category = model.predict(X_input)[0]
    return predicted_category


# -------------------------------
# Helper Functions
# -------------------------------

def get_expense_of_day(user):
    """Returns total expenses for the current day"""
    current_date = date.today()
    expenses = Expense.objects.filter(owner=user, date=current_date)
    return sum(expense.amount for expense in expenses)


# -------------------------------
# Views
# -------------------------------

@login_required(login_url='/authentication/login')
def search_expenses(request):
    if request.method == 'POST':
        search_str = json.loads(request.body).get('searchText')
        expenses = Expense.objects.filter(
            amount__istartswith=search_str, owner=request.user
        ) | Expense.objects.filter(
            date__istartswith=search_str, owner=request.user
        ) | Expense.objects.filter(
            description__icontains=search_str, owner=request.user
        ) | Expense.objects.filter(
            category__icontains=search_str, owner=request.user
        )
        data = expenses.values()
        return JsonResponse(list(data), safe=False)


@login_required(login_url='/authentication/login')
def index(request):
    categories = Category.objects.all()
    expenses = Expense.objects.filter(owner=request.user)

    sort_order = request.GET.get('sort')
    if sort_order == 'amount_asc':
        expenses = expenses.order_by('amount')
    elif sort_order == 'amount_desc':
        expenses = expenses.order_by('-amount')
    elif sort_order == 'date_asc':
        expenses = expenses.order_by('date')
    elif sort_order == 'date_desc':
        expenses = expenses.order_by('-date')

    paginator = Paginator(expenses, 5)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    try:
        currency = UserPreference.objects.get(user=request.user).currency
    except UserPreference.DoesNotExist:
        currency = None

    total = page_obj.paginator.num_pages
    context = {
        'expenses': expenses,
        'page_obj': page_obj,
        'currency': currency,
        'total': total,
        'sort_order': sort_order,
    }
    return render(request, 'expenses/index.html', context)


@login_required(login_url='/authentication/login')
def add_expense(request):
    categories = Category.objects.all()
    context = {'categories': categories, 'values': request.POST}

    if request.method == 'GET':
        return render(request, 'expenses/add_expense.html', context)

    if request.method == 'POST':
        amount = request.POST.get('amount')
        description = request.POST.get('description')
        date_str = request.POST.get('expense_date')
        predicted_category = request.POST.get('category')
        initial_predicted_category = request.POST.get('initial_predicted_category')

        if not amount:
            messages.error(request, 'Amount is required')
            return render(request, 'expenses/add_expense.html', context)
        if not description:
            messages.error(request, 'Description is required')
            return render(request, 'expenses/add_expense.html', context)

        # Update dataset if user changed predicted category
        if predicted_category != initial_predicted_category:
            new_data = {'description': description, 'category': predicted_category}
            update_url = 'http://127.0.0.1:8000/api/update-dataset/'
            requests.post(update_url, json={'new_data': new_data})

        try:
            expense_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            today = date.today()
            if expense_date > today:
                messages.error(request, 'Date cannot be in the future')
                return render(request, 'expenses/add_expense.html', context)

            user = request.user
            daily_limit = ExpenseLimit.objects.filter(owner=user).first()
            daily_limit_value = daily_limit.daily_expense_limit if daily_limit else 5000

            total_today = get_expense_of_day(user) + float(amount)
            if total_today > daily_limit_value:
                subject = 'Daily Expense Limit Exceeded'
                message = f'Hello {user.username},\n\nYour expenses today exceed your daily limit.'
                send_mail(subject, message, settings.EMAIL_HOST_USER, [user.email])
                messages.warning(request, 'Your expenses for today exceed your daily limit')

            Expense.objects.create(
                owner=user,
                amount=amount,
                date=expense_date,
                category=predicted_category,
                description=description
            )
            messages.success(request, 'Expense saved successfully')
            return redirect('expenses')
        except ValueError:
            messages.error(request, 'Invalid date format')
            return render(request, 'expenses/add_expense.html', context)


@login_required(login_url='/authentication/login')
def expense_edit(request, id):
    expense = Expense.objects.get(pk=id)
    categories = Category.objects.all()
    context = {'expense': expense, 'values': expense, 'categories': categories}

    if request.method == 'GET':
        return render(request, 'expenses/edit-expense.html', context)

    if request.method == 'POST':
        amount = request.POST.get('amount')
        description = request.POST.get('description')
        date_str = request.POST.get('expense_date')
        category = request.POST.get('category')

        if not amount or not description:
            messages.error(request, 'All fields are required')
            return render(request, 'expenses/edit-expense.html', context)

        try:
            expense_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            today = date.today()
            if expense_date > today:
                messages.error(request, 'Date cannot be in the future')
                return render(request, 'expenses/edit-expense.html', context)

            expense.amount = amount
            expense.date = expense_date
            expense.category = category
            expense.description = description
            expense.save()
            messages.success(request, 'Expense updated successfully')
            return redirect('expenses')
        except ValueError:
            messages.error(request, 'Invalid date format')
            return render(request, 'expenses/edit-expense.html', context)


@login_required(login_url='/authentication/login')
def delete_expense(request, id):
    Expense.objects.get(pk=id).delete()
    messages.success(request, 'Expense removed')
    return redirect('expenses')


@login_required(login_url='/authentication/login')
def expense_category_summary(request):
    today = date.today()
    six_months_ago = today - datetime.timedelta(days=30*6)
    expenses = Expense.objects.filter(owner=request.user, date__gte=six_months_ago, date__lte=today)
    categories = set(exp.category for exp in expenses)
    finalrep = {cat: sum(exp.amount for exp in expenses.filter(category=cat)) for cat in categories}
    return JsonResponse({'expense_category_data': finalrep}, safe=False)


@login_required(login_url='/authentication/login')
def stats_view(request):
    return render(request, 'expenses/stats.html')


@login_required(login_url='/authentication/login')
def set_expense_limit(request):
    if request.method == "POST":
        daily_limit = request.POST.get('daily_expense_limit')
        existing = ExpenseLimit.objects.filter(owner=request.user).first()
        if existing:
            existing.daily_expense_limit = daily_limit
            existing.save()
        else:
            ExpenseLimit.objects.create(owner=request.user, daily_expense_limit=daily_limit)
        messages.success(request, "Daily Expense Limit Updated Successfully!")
    return HttpResponseRedirect('/preferences/')
